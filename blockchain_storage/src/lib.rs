//! Application-level interface for the persistent blockchain storage.

use common::chain::block::Block;
use common::chain::transaction::{Transaction, TxMainChainIndex, TxMainChainPosition};
use common::primitives::{BlockHeight, Id};

//#[cfg(any(test, feature = "mock"))]
//pub mod mock;
mod store;

pub use store::Store;

/// Blockchain storage error
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, thiserror::Error)]
pub enum Error {
    #[error("Storage error: {0}")]
    Storage(storage::error::Recoverable),
}

impl From<storage::Error> for Error {
    fn from(e: storage::Error) -> Self {
        Error::Storage(e.recoverable())
    }
}

/// Possibly failing result of blockchain storage query
pub type Result<T> = core::result::Result<T, Error>;

/// Queries on persistent blockchain data
pub trait BlockchainStorageRead<'a> {
    /// Get storage version
    fn get_storage_version(&'a self) -> crate::Result<u32>;

    /// Get the hash of the best block
    fn get_best_block_id(&'a self) -> crate::Result<Option<Id<Block>>>;

    /// Get block by its hash
    fn get_block(&'a self, id: Id<Block>) -> crate::Result<Option<Block>>;

    /// Get outputs state for given transaction in the mainchain
    fn get_mainchain_tx_index(
        &'a self,
        tx_id: &Id<Transaction>,
    ) -> crate::Result<Option<TxMainChainIndex>>;

    /// Get transaction by block ID and position
    fn get_mainchain_tx_by_position(
        &'a self,
        tx_index: &TxMainChainPosition,
    ) -> crate::Result<Option<Transaction>>;

    /// Get transaction by transaction ID. Transaction must be in the index.
    fn get_mainchain_tx(&'a self, txid: &Id<Transaction>) -> crate::Result<Option<Transaction>>;

    /// Get mainchain block by its height
    fn get_block_id_by_height(&'a self, height: &BlockHeight) -> crate::Result<Option<Id<Block>>>;
}

/// Modifying operations on persistent blockchain data
pub trait BlockchainStorageWrite<'a>: BlockchainStorageRead<'a> {
    /// Set storage version
    fn set_storage_version(&'a mut self, version: u32) -> crate::Result<()>;

    /// Set the hash of the best block
    fn set_best_block_id(&'a mut self, id: &Id<Block>) -> crate::Result<()>;

    /// Add a new block into the database
    fn add_block(&'a mut self, block: &Block) -> crate::Result<()>;

    /// Remove block from the database
    fn del_block(&'a mut self, id: Id<Block>) -> crate::Result<()>;

    /// Set state of the outputs of given transaction
    fn set_mainchain_tx_index(
        &'a mut self,
        tx_id: &Id<Transaction>,
        tx_index: &TxMainChainIndex,
    ) -> crate::Result<()>;

    /// Delete outputs state index associated with given transaction
    fn del_mainchain_tx_index(&'a mut self, tx_id: &Id<Transaction>) -> crate::Result<()>;

    /// Set the mainchain block at given height to be given block.
    fn set_block_id_at_height(
        &'a mut self,
        height: &BlockHeight,
        block_id: &Id<Block>,
    ) -> crate::Result<()>;

    /// Remove block id from given mainchain height
    fn del_block_id_at_height(&'a mut self, height: &BlockHeight) -> crate::Result<()>;
}
