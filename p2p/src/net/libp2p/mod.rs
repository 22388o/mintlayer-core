// Copyright (c) 2021 Protocol Labs
// Copyright (c) 2021-2022 RBB S.r.l
// opensource@mintlayer.org
// SPDX-License-Identifier: MIT
// Licensed under the MIT License;
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://spdx.org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author(s): A. Altonen
use crate::{
    error::{self, Libp2pError, P2pError},
    net::{self, Event, GossipSubTopic, NetworkService, SocketService},
};
use async_trait::async_trait;
use futures::prelude::*;
use itertools::*;
use libp2p::{
    core::{upgrade, PeerId},
    identity,
    mdns::Mdns,
    mplex,
    multiaddr::Protocol,
    noise,
    streaming::{IdentityCodec, StreamHandle, Streaming},
    swarm::{NegotiatedSubstream, SwarmBuilder},
    tcp::TcpConfig,
    Multiaddr, Transport,
};
use logging::log;
use parity_scale_codec::{Decode, Encode};
use std::sync::Arc;
use tokio::sync::{
    mpsc::{Receiver, Sender},
    oneshot,
};

mod backend;
mod common;

// Maximum message size of 10 MB
const MESSAGE_MAX_SIZE: u32 = 10 * 1024 * 1024;

/// libp2p-specifc peer discovery strategies
#[derive(Debug, PartialEq, Eq)]
pub enum Libp2pStrategy {
    /// Use mDNS to find peers in the local network
    MulticastDns,
}

#[derive(Debug)]
pub struct Libp2pService {
    /// Multiaddress of the local peer
    pub addr: Multiaddr,

    /// TX channel for sending commands to libp2p backend
    cmd_tx: Sender<common::Command>,

    /// RX channel for receiving events from libp2p backend
    event_rx: Receiver<common::Event>,
}

#[derive(Debug)]
#[allow(unused)]
pub struct Libp2pSocket {
    /// Multiaddress of the remote peer
    addr: Multiaddr,

    /// Unique ID of the peer
    id: PeerId,

    /// Stream handle for the remote peer
    stream: StreamHandle<NegotiatedSubstream>,
}

/// Verify that the discovered multiaddress is in a format that Mintlayer supports:
///   /ip4/<IPv4 address>/tcp/<TCP port>/p2p/<PeerId multihash>
///   /ip6/<IPv6 address>/tcp/<TCP port>/p2p/<PeerId multihash>
///
/// Documentation for libp2p-mdns doesn't mention if `peer_addr` includes the PeerId
/// so if it doesn't, add it. Otherwise just return the address
fn parse_discovered_addr(peer_id: PeerId, peer_addr: Multiaddr) -> Option<Multiaddr> {
    let mut components = peer_addr.iter();

    if !std::matches!(
        components.next(),
        Some(Protocol::Ip4(_)) | Some(Protocol::Ip6(_))
    ) {
        return None;
    }

    if !std::matches!(components.next(), Some(Protocol::Tcp(_))) {
        return None;
    }

    match components.next() {
        Some(Protocol::P2p(_)) => Some(peer_addr.clone()),
        None => Some(peer_addr.with(Protocol::P2p(peer_id.into()))),
        Some(_) => None,
    }
}

/// Get the network layer protocol from `addr`
fn get_addr_from_multiaddr(addr: &Multiaddr) -> Option<Protocol> {
    addr.iter().next()
}

impl<T> FromIterator<(PeerId, Multiaddr)> for net::AddrInfo<T>
where
    T: NetworkService<PeerId = PeerId, Address = Multiaddr>,
{
    fn from_iter<I: IntoIterator<Item = (PeerId, Multiaddr)>>(iter: I) -> Self {
        let mut entry = net::AddrInfo {
            id: PeerId::random(),
            ip4: Vec::new(),
            ip6: Vec::new(),
        };

        iter.into_iter().for_each(|(id, addr)| {
            entry.id = id;
            match get_addr_from_multiaddr(&addr) {
                Some(Protocol::Ip4(_)) => entry.ip4.push(Arc::new(addr)),
                Some(Protocol::Ip6(_)) => entry.ip6.push(Arc::new(addr)),
                _ => panic!("parse_discovered_addr() failed!"),
            }
        });
        log::trace!(
            "id {:?}, ipv4 {:#?}, ipv6 {:#?}",
            entry.id,
            entry.ip4,
            entry.ip6
        );

        entry
    }
}

/// Parse all discovered addresses and group them by PeerId
fn parse_peers<T>(mut peers: Vec<(PeerId, Multiaddr)>) -> Vec<net::AddrInfo<T>>
where
    T: NetworkService<PeerId = PeerId, Address = Multiaddr>,
{
    peers.sort_by(|a, b| a.0.cmp(&b.0));
    peers
        .into_iter()
        .map(|(id, addr)| (id, parse_discovered_addr(id, addr)))
        .filter(|(_id, addr)| addr.is_some())
        .map(|(id, addr)| (id, addr.unwrap()))
        .group_by(|info| info.0)
        .into_iter()
        .map(|(_id, addrs)| net::AddrInfo::from_iter(addrs))
        .collect::<Vec<net::AddrInfo<T>>>()
}

#[async_trait]
impl NetworkService for Libp2pService {
    type Address = Multiaddr;
    type Socket = Libp2pSocket;
    type Strategy = Libp2pStrategy;
    type PeerId = PeerId;

    async fn new(
        addr: Self::Address,
        strategies: &[Self::Strategy],
        _topics: &[GossipSubTopic],
    ) -> error::Result<Self> {
        let id_keys = identity::Keypair::generate_ed25519();
        let peer_id = id_keys.public().to_peer_id();
        let noise_keys = noise::Keypair::<noise::X25519Spec>::new().into_authentic(&id_keys)?;

        let transport = TcpConfig::new()
            .nodelay(true)
            .port_reuse(true)
            .upgrade(upgrade::Version::V1)
            .authenticate(noise::NoiseConfig::xx(noise_keys).into_authenticated())
            .multiplex(mplex::MplexConfig::new())
            .boxed();

        let swarm = SwarmBuilder::new(
            transport,
            common::ComposedBehaviour {
                streaming: Streaming::<IdentityCodec>::default(),
                mdns: Mdns::new(Default::default()).await?,
            },
            peer_id,
        )
        .build();

        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::channel(16);
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(16);

        // If mDNS has been specified as a peer discovery strategy for this Libp2pService,
        // pass that information to the backend so it knows to relay the mDNS events to P2P
        let relay_mdns = strategies.iter().any(|s| s == &Libp2pStrategy::MulticastDns);
        log::trace!("multicast dns enabled {}", relay_mdns);

        // run the libp2p backend in a background task
        log::debug!("spawning libp2p backend to background");

        tokio::spawn(async move {
            let mut backend = backend::Backend::new(swarm, cmd_rx, event_tx, relay_mdns);
            backend.run().await
        });

        // send listen command to the libp2p backend and if it succeeds,
        // create a multiaddress for local peer and return the Libp2pService object
        let (tx, rx) = oneshot::channel();
        cmd_tx
            .send(common::Command::Listen {
                addr: addr.clone(),
                response: tx,
            })
            .await?;
        rx.await?.map_err(|_| P2pError::SocketError(std::io::ErrorKind::AddrInUse))?;

        Ok(Self {
            addr: addr.with(Protocol::P2p(peer_id.into())),
            cmd_tx,
            event_rx,
        })
    }

    async fn connect(
        &mut self,
        addr: Self::Address,
    ) -> error::Result<(Self::PeerId, Self::Socket)> {
        log::trace!("try to establish outbound connection, address {:?}", addr);

        let peer_id = match addr.iter().last() {
            Some(Protocol::P2p(hash)) => PeerId::from_multihash(hash).map_err(|_| {
                P2pError::Libp2pError(Libp2pError::DialError(
                    "Expect peer multiaddr to contain peer ID.".into(),
                ))
            })?,
            _ => {
                return Err(P2pError::Libp2pError(Libp2pError::DialError(
                    "Expect peer multiaddr to contain peer ID.".into(),
                )))
            }
        };

        // dial the remote peer
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(common::Command::Dial {
                peer_id,
                peer_addr: addr.clone(),
                response: tx,
            })
            .await?;

        // wait for command response
        rx.await
            .map_err(|e| e)? // channel closed
            .map_err(|e| e)?; // command failure

        // if dial succeeded, open a generic stream
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(common::Command::OpenStream {
                peer_id,
                response: tx,
            })
            .await?;

        let stream = rx
            .await
            .map_err(|e| e)? // channel closed
            .map_err(|e| e)?; // command failure

        Ok((
            peer_id,
            Libp2pSocket {
                id: peer_id,
                addr,
                stream,
            },
        ))
    }

    async fn poll_next<T>(&mut self) -> error::Result<Event<T>>
    where
        T: NetworkService<Socket = Libp2pSocket, Address = Multiaddr, PeerId = PeerId>,
    {
        match self.event_rx.recv().await.ok_or(P2pError::ChannelClosed)? {
            common::Event::ConnectionAccepted { socket } => {
                Ok(Event::IncomingConnection(socket.id, *socket))
            }
            common::Event::PeerDiscovered { peers } => {
                Ok(Event::PeerDiscovered(parse_peers(peers)))
            }
            common::Event::PeerExpired { peers } => Ok(Event::PeerExpired(parse_peers(peers))),
        }
    }

    async fn publish<T>(&mut self, _topic: GossipSubTopic, _data: &T)
    where
        T: Sync + Send + Encode,
    {
        todo!();
    }
}

#[async_trait]
impl SocketService for Libp2pSocket {
    async fn send<T>(&mut self, data: &T) -> error::Result<()>
    where
        T: Sync + Send + Encode,
    {
        let encoded = data.encode();
        let size = (encoded.len() as u32).encode();

        log::trace!("try to send message, {} bytes", encoded.len());

        self.stream
            .write_all(&size)
            .await
            .map_err(|e| P2pError::SocketError(e.kind()))?;

        self.stream
            .write_all(&encoded)
            .await
            .map_err(|e| P2pError::SocketError(e.kind()))?;

        self.stream.flush().await.map_err(|e| P2pError::SocketError(e.kind()))
    }

    async fn recv<T>(&mut self) -> error::Result<T>
    where
        T: Decode,
    {
        let mut size: u32 = 0u32;
        let mut data = vec![0u8; size.encoded_size()];

        size = match self.stream.read_exact(&mut data).await {
            Ok(_) => Decode::decode(&mut &data[..])
                .map_err(|e| P2pError::DecodeFailure(e.to_string()))?,
            Err(_) => return Err(P2pError::PeerDisconnected),
        };
        log::trace!("try to read message, {} bytes", size);

        if size > MESSAGE_MAX_SIZE {
            return Err(P2pError::DecodeFailure("Message is too big".to_string()));
        }
        data.resize(size as usize, 0);

        match self.stream.read_exact(&mut data).await {
            Ok(_) => Decode::decode(&mut &data[..]).map_err(|e| e.into()),
            Err(_) => return Err(P2pError::PeerDisconnected),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net;

    #[derive(Debug, Encode, Decode, PartialEq, Eq, Copy, Clone)]
    struct Transaction {
        hash: u64,
        value: u128,
    }

    #[tokio::test]
    async fn test_connect_new() {
        let service = Libp2pService::new("/ip6/::1/tcp/8900".parse().unwrap(), &[], &[]).await;
        assert!(service.is_ok());
    }

    // verify that binding to the same interface twice is not possible
    #[ignore]
    #[tokio::test]
    async fn test_connect_new_addrinuse() {
        let service = Libp2pService::new("/ip6/::1/tcp/8901".parse().unwrap(), &[], &[]).await;
        assert!(service.is_ok());

        let service = Libp2pService::new("/ip6/::1/tcp/8901".parse().unwrap(), &[], &[]).await;

        match service {
            Err(e) => {
                assert_eq!(e, P2pError::SocketError(std::io::ErrorKind::AddrInUse));
            }
            Ok(_) => panic!("address is not in use"),
        }
    }

    // try to connect two nodes together by having `service1` listen for network events
    // and having `service2` trying to connect to `service1`
    #[tokio::test]
    async fn test_connect_accept() {
        let service1 = Libp2pService::new("/ip6/::1/tcp/8902".parse().unwrap(), &[], &[]).await;
        let service2 = Libp2pService::new("/ip6/::1/tcp/8903".parse().unwrap(), &[], &[]).await;
        assert!(service1.is_ok());
        assert!(service2.is_ok());

        let mut service1 = service1.unwrap();
        let mut service2 = service2.unwrap();
        let conn_addr = service1.addr.clone();

        let (res1, res2): (error::Result<Event<Libp2pService>>, _) =
            tokio::join!(service1.poll_next(), service2.connect(conn_addr));

        assert!(res2.is_ok());
        assert!(res1.is_ok());
    }

    // try to connect to a remote peer with a multiaddress that's missing the peerid
    // and verify that the connection fails
    #[tokio::test]
    async fn test_connect_peer_id_missing() {
        let addr1: Multiaddr = "/ip6/::1/tcp/8904".parse().unwrap();
        let mut service2 = Libp2pService::new("/ip6/::1/tcp/8905".parse().unwrap(), &[], &[])
            .await
            .unwrap();
        match service2.connect(addr1).await {
            Ok(_) => panic!("connect succeeded without peer id"),
            Err(e) => {
                assert_eq!(
                    e,
                    P2pError::Libp2pError(Libp2pError::DialError(
                        "Expect peer multiaddr to contain peer ID.".into(),
                    ))
                )
            }
        }
    }

    // connect two libp2p services together and send a transaction from one service
    // to another and verify that the transaction was received successfully and that
    // it decodes to the same transaction that was sent.
    #[tokio::test]
    async fn test_peer_send() {
        let service1 = Libp2pService::new("/ip6/::1/tcp/8905".parse().unwrap(), &[], &[]).await;
        let service2 = Libp2pService::new("/ip6/::1/tcp/8906".parse().unwrap(), &[], &[]).await;

        let mut service1 = service1.unwrap();
        let mut service2 = service2.unwrap();
        let conn_addr = service1.addr.clone();

        let (res1, res2): (error::Result<Event<Libp2pService>>, _) =
            tokio::join!(service1.poll_next(), service2.connect(conn_addr));

        let mut socket1 = match res1.unwrap() {
            net::Event::IncomingConnection(_, socket) => socket,
            _ => panic!("invalid event received, expected incoming connection"),
        };
        let mut socket2 = res2.unwrap().1;

        // try to send data that implements `Encode + Decode`
        // and verify that it was received correctly
        let tx = Transaction {
            hash: u64::MAX,
            value: u128::MAX,
        };
        let encoded_size: u32 = tx.encode().len() as u32;

        let mut buf = vec![0u8; 64];
        let (server_res, peer_res) = tokio::join!(socket2.stream.read(&mut buf), socket1.send(&tx));

        assert!(peer_res.is_ok());
        assert!(server_res.is_ok());

        let received_size: u32 = Decode::decode(&mut &buf[..]).unwrap();
        assert_eq!(received_size, encoded_size);

        buf.resize(received_size as usize, 0);
        socket2.stream.read_exact(&mut buf).await.unwrap();
        assert_eq!(Decode::decode(&mut &buf[..]), Ok(tx));
    }

    // connect two libp2p services together and send a transaction from one service
    // to another and verify that the transaction was received successfully and that
    // it decodes to the same transaction that was sent.
    #[tokio::test]
    async fn test_peer_recv() {
        let service1 = Libp2pService::new("/ip6/::1/tcp/8907".parse().unwrap(), &[], &[]).await;
        let service2 = Libp2pService::new("/ip6/::1/tcp/8908".parse().unwrap(), &[], &[]).await;

        let mut service1 = service1.unwrap();
        let mut service2 = service2.unwrap();
        let conn_addr = service1.addr.clone();

        let (res1, res2): (error::Result<Event<Libp2pService>>, _) =
            tokio::join!(service1.poll_next(), service2.connect(conn_addr));

        let mut socket1 = match res1.unwrap() {
            net::Event::IncomingConnection(_, socket) => socket,
            _ => panic!("invalid event received, expected incoming connection"),
        };
        let mut socket2 = res2.unwrap().1;

        let tx = Transaction {
            hash: u64::MAX,
            value: u128::MAX,
        };
        let tx_copy = tx;

        let (res1, res2): (_, Result<Transaction, _>) =
            tokio::join!(socket2.send(&tx_copy), socket1.recv());

        assert!(res1.is_ok());
        assert!(res2.is_ok());
        assert_eq!(res2.unwrap(), tx);
    }

    // connect two libp2p services together and send multiple transactions from
    // one service to another and verify that they are buffered in the receiving
    // end and decoded as three separate transactions
    #[tokio::test]
    async fn test_peer_buffered_recv() {
        let service1 = Libp2pService::new("/ip6/::1/tcp/8909".parse().unwrap(), &[], &[]).await;
        let service2 = Libp2pService::new("/ip6/::1/tcp/8910".parse().unwrap(), &[], &[]).await;

        let mut service1 = service1.unwrap();
        let mut service2 = service2.unwrap();
        let conn_addr = service1.addr.clone();

        let (res1, res2): (error::Result<Event<Libp2pService>>, _) =
            tokio::join!(service1.poll_next(), service2.connect(conn_addr));

        let mut socket1 = match res1.unwrap() {
            net::Event::IncomingConnection(_, socket) => socket,
            _ => panic!("invalid event received, expected incoming connection"),
        };
        let mut socket2 = res2.unwrap().1;

        let tx = Transaction {
            hash: u64::MAX,
            value: u128::MAX,
        };
        let tx_copy = tx;

        for _ in 0..3 {
            assert_eq!(socket2.send(&tx_copy).await, Ok(()));
        }

        for _ in 0..3 {
            let res: Result<Transaction, _> = socket1.recv().await;
            assert!(res.is_ok());
            assert_eq!(res.unwrap(), tx);
        }
    }

    // connect two libp2p services together and try to send a message
    // that is too big and verify that it is rejected
    #[tokio::test]
    async fn test_too_large_message_size() {
        let service1 = Libp2pService::new("/ip6/::1/tcp/8911".parse().unwrap(), &[], &[]).await;
        let service2 = Libp2pService::new("/ip6/::1/tcp/8912".parse().unwrap(), &[], &[]).await;

        let mut service1 = service1.unwrap();
        let mut service2 = service2.unwrap();
        let conn_addr = service1.addr.clone();

        let (res1, res2): (error::Result<Event<Libp2pService>>, _) =
            tokio::join!(service1.poll_next(), service2.connect(conn_addr));

        let mut socket1 = match res1.unwrap() {
            net::Event::IncomingConnection(_, socket) => socket,
            _ => panic!("invalid event received, expected incoming connection"),
        };
        let mut socket2 = res2.unwrap().1;

        // send a message size of 4GB to remote
        let msg_size = u32::MAX.encode();
        socket1.stream.write_all(&msg_size).await.unwrap();
        socket1.stream.flush().await.unwrap();

        let res: Result<Transaction, _> = socket2.recv().await;
        assert_eq!(
            res,
            Err(P2pError::DecodeFailure("Message is too big".to_string()))
        );
    }

    #[test]
    fn test_parse_discovered_addr() {
        let peer_id: PeerId =
            "12D3KooWE3kBRAnn6jxZMdK1JMWx1iHtR1NKzXSRv5HLTmfD9u9c".parse().unwrap();

        assert_eq!(
            parse_discovered_addr(peer_id, "/ip4/127.0.0.1/udp/9090/quic".parse().unwrap()),
            None
        );
        assert_eq!(
            parse_discovered_addr(peer_id, "/ip6/::1/udp/3217".parse().unwrap()),
            None
        );
        assert_eq!(
            parse_discovered_addr(peer_id, "/ip4/127.0.0.1/tcp/9090/quic".parse().unwrap()),
            None
        );
        assert_eq!(
            parse_discovered_addr(peer_id, "/ip4/127.0.0.1/tcp/80/http".parse().unwrap()),
            None
        );
        assert_eq!(
            parse_discovered_addr(peer_id, "/dns4/foo.com/tcp/80/http".parse().unwrap()),
            None
        );
        assert_eq!(
            parse_discovered_addr(peer_id, "/dns6/foo.com/tcp/443/https".parse().unwrap()),
            None
        );

        let addr: Multiaddr =
            "/ip6/::1/tcp/3217/p2p/12D3KooWRn14SemPVxwzdQNg8e8Trythiww1FWrNfPbukYBmZEbJ"
                .parse()
                .unwrap();
        let id: PeerId = "12D3KooWRn14SemPVxwzdQNg8e8Trythiww1FWrNfPbukYBmZEbJ".parse().unwrap();
        assert_eq!(parse_discovered_addr(id, addr.clone()), Some(addr));

        let id: PeerId = "12D3KooWRn14SemPVxwzdQNg8e8Trythiww1FWrNfPbukYBmZEbJ".parse().unwrap();
        let addr: Multiaddr =
            "/ip4/127.0.0.1/tcp/9090/p2p/12D3KooWRn14SemPVxwzdQNg8e8Trythiww1FWrNfPbukYBmZEbJ"
                .parse()
                .unwrap();
        assert_eq!(parse_discovered_addr(id, addr.clone()), Some(addr));

        let id: PeerId = "12D3KooWRn14SemPVxwzdQNg8e8Trythiww1FWrNfPbukYBmZEbJ".parse().unwrap();
        let addr: Multiaddr = "/ip6/::1/tcp/3217".parse().unwrap();
        assert_eq!(
            parse_discovered_addr(id, addr.clone()),
            Some(addr.with(Protocol::P2p(id.into())))
        );

        let id: PeerId = "12D3KooWRn14SemPVxwzdQNg8e8Trythiww1FWrNfPbukYBmZEbJ".parse().unwrap();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/9090".parse().unwrap();
        assert_eq!(
            parse_discovered_addr(id, addr.clone()),
            Some(addr.with(Protocol::P2p(id.into())))
        );
    }

    impl PartialEq for Libp2pService {
        fn eq(&self, other: &Self) -> bool {
            self.addr == other.addr
        }
    }

    // verify that vector of address (that all belong to one peer) parse into one `net::Peer` entry
    #[test]
    fn test_parse_peers_valid_1_peer() {
        let id: PeerId = "12D3KooWRn14SemPVxwzdQNg8e8Trythiww1FWrNfPbukYBmZEbJ".parse().unwrap();
        let ip4: Multiaddr = "/ip4/127.0.0.1/tcp/9090".parse().unwrap();
        let ip6: Multiaddr = "/ip6/::1/tcp/9091".parse().unwrap();
        let addrs = vec![(id, ip4.clone()), (id, ip6.clone())];

        let parsed: Vec<net::AddrInfo<Libp2pService>> = parse_peers(addrs);
        assert_eq!(
            parsed,
            vec![net::AddrInfo {
                id,
                ip4: vec![Arc::new(ip4.with(Protocol::P2p(id.into())))],
                ip6: vec![Arc::new(ip6.with(Protocol::P2p(id.into())))],
            }]
        );
    }

    // discovery 5 different addresses, ipv4 and ipv6 for both peer and an additional
    // dns address for peer
    //
    // verify that `parse_peers` returns two peers and both only have ipv4 and ipv6 addresses
    #[test]
    fn test_parse_peers_valid_2_peers() {
        let id_1: PeerId = "12D3KooWRn14SemPVxwzdQNg8e8Trythiww1FWrNfPbukYBmZEbJ".parse().unwrap();
        let ip4_1: Multiaddr = "/ip4/127.0.0.1/tcp/9090".parse().unwrap();
        let ip6_1: Multiaddr = "/ip6/::1/tcp/9091".parse().unwrap();

        let id_2: PeerId = "12D3KooWE3kBRAnn6jxZMdK1JMWx1iHtR1NKzXSRv5HLTmfD9u9c".parse().unwrap();
        let ip4_2: Multiaddr = "/ip4/127.0.0.1/tcp/8080".parse().unwrap();
        let ip6_2: Multiaddr = "/ip6/::1/tcp/8081".parse().unwrap();
        let dns: Multiaddr = "/dns4/foo.com/tcp/80/http".parse().unwrap();

        let addrs = vec![
            (id_1, ip4_1.clone()),
            (id_2, ip4_2.clone()),
            (id_2, ip6_2.clone()),
            (id_1, ip6_1.clone()),
            (id_2, dns),
        ];

        let mut parsed: Vec<net::AddrInfo<Libp2pService>> = parse_peers(addrs);
        parsed.sort_by(|a, b| a.id.cmp(&b.id));

        assert_eq!(
            parsed,
            vec![
                net::AddrInfo {
                    id: id_2,
                    ip4: vec![Arc::new(ip4_2.with(Protocol::P2p(id_2.into())))],
                    ip6: vec![Arc::new(ip6_2.with(Protocol::P2p(id_2.into())))],
                },
                net::AddrInfo {
                    id: id_1,
                    ip4: vec![Arc::new(ip4_1.with(Protocol::P2p(id_1.into())))],
                    ip6: vec![Arc::new(ip6_1.with(Protocol::P2p(id_1.into())))],
                },
            ]
        );
    }

    // find 3 peers but only one of the peers have an accepted address available so verify
    // that `parse_peers()` returns only that peer
    #[test]
    fn test_parse_peers_valid_3_peers_1_valid() {
        let id_1: PeerId = "12D3KooWRn14SemPVxwzdQNg8e8Trythiww1FWrNfPbukYBmZEbJ".parse().unwrap();
        let ip4: Multiaddr = "/ip4/127.0.0.1/tcp/9090".parse().unwrap();

        let id_2: PeerId = "12D3KooWE3kBRAnn6jxZMdK1JMWx1iHtR1NKzXSRv5HLTmfD9u9c".parse().unwrap();
        let dns: Multiaddr = "/dns4/foo.com/tcp/80/http".parse().unwrap();

        let id_3: PeerId = "12D3KooWGK4RzvNeioS9aXdzmYXU3mgDrRPjQd8SVyXCkHNxLbWN".parse().unwrap();
        let quic: Multiaddr = "/ip4/127.0.0.1/tcp/9090/quic".parse().unwrap();

        let addrs = vec![(id_1, ip4.clone()), (id_2, dns), (id_3, quic)];
        let parsed: Vec<net::AddrInfo<Libp2pService>> = parse_peers(addrs);

        assert_eq!(
            parsed,
            vec![net::AddrInfo {
                id: id_1,
                ip4: vec![Arc::new(ip4.with(Protocol::P2p(id_1.into())))],
                ip6: vec![],
            }]
        );
    }
}
