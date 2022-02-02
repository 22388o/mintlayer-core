use crate::chain::transaction::Transaction;
use crate::primitives::Id;
use parity_scale_codec::{Decode, Encode};

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct OutPoint {
    id: Id<Transaction>,
    index: u32,
}

impl OutPoint {
    pub const COINBASE_OUTPOINT_INDEX: u32 = u32::MAX;

    pub fn new(prev_tx_id: Id<Transaction>, output_index: u32) -> Self {
        OutPoint {
            id: prev_tx_id,
            index: output_index,
        }
    }

    pub fn get_tx_id(&self) -> Id<Transaction> {
        self.id.clone()
    }

    pub fn get_output_index(&self) -> u32 {
        self.index
    }

    pub fn is_coinbase(&self) -> bool {
        self.id.is_null() && self.index == Self::COINBASE_OUTPOINT_INDEX
    }
}

impl PartialOrd for OutPoint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id.get().partial_cmp(&other.id.get()).and_then(|ordering| match ordering {
            std::cmp::Ordering::Equal => self.index.partial_cmp(&other.index),
            _ => Some(ordering),
        })
    }
}

impl Ord for OutPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let id_ord = self.id.get().cmp(&other.id.get());
        match id_ord {
            std::cmp::Ordering::Equal => self.index.cmp(&other.index),
            _ => id_ord,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct TxInput {
    outpoint: OutPoint,
    witness: Vec<u8>,
}

impl TxInput {
    pub fn new(prev_tx_id: Id<Transaction>, output_index: u32, witness: Vec<u8>) -> Self {
        TxInput {
            outpoint: OutPoint::new(prev_tx_id, output_index),
            witness,
        }
    }

    pub fn get_outpoint(&self) -> &OutPoint {
        &self.outpoint
    }

    pub fn get_witness(&self) -> &Vec<u8> {
        &self.witness
    }
}
