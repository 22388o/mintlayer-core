use std::cmp::Ord;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;

use parity_scale_codec::Encode;
use thiserror::Error;

use common::chain::transaction::Transaction;
use common::chain::transaction::TxInput;
use common::chain::OutPoint;
use common::primitives::amount::Amount;
use common::primitives::Id;
use common::primitives::Idable;
use common::primitives::H256;
// TODO Add a newtype macro

// TODO this willbe defined elsewhere
const MAX_BLOCK_SIZE: usize = 1_000_000;

pub const MEMPOOL_MAX_TXS: usize = 1_000_000;

#[derive(PartialEq, Eq, Clone, Hash)]
struct TxMemPoolEntry {
    tx: Arc<Transaction>,
    parents: BTreeSet<TxMemPoolEntry>,
}

// TODO consider mocking the fees more accurately by referencing here a global mock Chain State
// object
fn compare_fees(this: &TxMemPoolEntry, other: &TxMemPoolEntry) -> Ordering {
    // We want the comparison to be such that a collection is sorted if its elements are ordered in
    // descending order
    other.get_fee().cmp(&this.get_fee())
}

impl TxMemPoolEntry {
    //TODO this should really be sum of inputs minus sum of outputs
    //But we don't yet have a way of summing of the inputs
    //TODO handle the error bettter
    fn get_fee(&self) -> Amount {
        self.tx
            .get_outputs()
            .iter()
            .map(|output| output.get_value())
            .sum::<Option<_>>()
            .expect("fee overflow")
    }
}

impl From<Arc<Transaction>> for TxMemPoolEntry {
    fn from(tx: Arc<Transaction>) -> Self {
        Self {
            tx,
            parents: BTreeSet::default(),
        }
    }
}

impl AsRef<Transaction> for TxMemPoolEntry {
    fn as_ref(&self) -> &Transaction {
        &self.tx
    }
}

impl PartialOrd for TxMemPoolEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(compare_fees(self, other))
    }
}

impl Ord for TxMemPoolEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_fees(self, other)
    }
}

pub(crate) struct MempoolImpl<C: ChainStateView> {
    store: MempoolStore,
    chain_state: C,
}

struct MempoolStore {
    // TODO some cloning can be avoided by having spender_txs be a map between references
    txs_by_fee: BTreeSet<TxMemPoolEntry>,
    txs_by_id: HashMap<H256, TxMemPoolEntry>,
    spender_txs: HashMap<OutPoint, Arc<Transaction>>,
}

impl MempoolStore {
    fn new() -> Self {
        Self {
            txs_by_fee: BTreeSet::new(),
            txs_by_id: HashMap::new(),
            spender_txs: HashMap::new(),
        }
    }

    // TODO do we have a limit on the number of inputs per TX?

    // Checks whether the outpoint belongs to one of the txs in the mempool
    fn contains_outpoint(&self, outpoint: &OutPoint) -> bool {
        matches!(self.txs_by_id.get(&outpoint.get_tx_id().get()),
            Some(entry) if entry.tx.get_inputs().len() > outpoint.get_output_index() as usize)
    }

    fn add_transaction(&mut self, tx: Transaction) -> Result<(), MempoolError> {
        let tx = Arc::new(tx);
        let mempool_tx = TxMemPoolEntry::from(tx.clone());
        self.txs_by_fee.insert(mempool_tx.clone());
        self.txs_by_id.insert(mempool_tx.as_ref().get_id().get(), mempool_tx);

        for outpoint in tx.get_inputs().iter().map(|input| input.get_outpoint()) {
            self.spender_txs.insert(outpoint.clone(), tx.clone());
        }
        // TODO introduce parent data structure and update it ihere
        Ok(())
    }

    fn find_conflicting_tx(&self, tx: &OutPoint) -> Option<Arc<Transaction>> {
        self.spender_txs.get(tx).cloned()
    }

    // TODO Make this return an iterator?
    // TODO see if can get rid of clone
    // TODO returning a HashSet requires that Transaction implement the Hash trait. If we return
    // something else (i.e. BTreeSet) we can get away with out this derive, but some member types
    // of Transaction will still require the derive
    fn ancestors(&self, entry: &TxMemPoolEntry) -> HashSet<TxMemPoolEntry> {
        let parents = entry.parents.clone();

        parents
            .clone()
            .into_iter()
            .chain(parents.into_iter().map(|parent| self.ancestors(&parent)).flatten())
            .collect::<_>()
    }
}

#[derive(Debug, Error)]
pub enum MempoolError {
    #[error("Mempool is full")]
    MempoolFull,
    #[error("GenericError")]
    GenericError,
    #[error(transparent)]
    TxValidationError(TxValidationError),
}

#[derive(Debug, Error)]
pub enum TxValidationError {
    #[error("No Inputs")]
    NoInputs,
    #[error("No Ouputs")]
    NoOutputs,
    #[error("DuplicateInputs")]
    DuplicateInputs,
    #[error("LooseCoinbase")]
    LooseCoinbase,
    #[error("OutPointNotFound")]
    OutPointNotFound { outpoint: OutPoint, tx: Transaction },
    #[error("ExceedsMaxBlockSize")]
    ExceedsMaxBlockSize,
    #[error("TransactionAlreadyInMempool")]
    TransactionAlreadyInMempool,
    #[error("ConflictWithIrreplaceableTransaction")]
    ConflictWithIrreplaceableTransaction,
}

impl From<TxValidationError> for MempoolError {
    fn from(e: TxValidationError) -> Self {
        MempoolError::TxValidationError(e)
    }
}

// We split the ChainState interface into two parts:
// 1. The "read-only" part of the interface, ChainStateView, is accessible to both the mempool (to validate transactoins) and the tests (to create mock transactions).
// 2. The "write" part, ApplyTx, is exposed ONLY to the mempool, not the tests. This is to prevent tests from modifying the ChainState object directly
pub trait ChainState: ChainStateView + ApplyTx {}

pub trait ChainStateView {
    fn contains_outpoint(&self, outpoint: &OutPoint) -> bool;
    fn get_outpoint_value(&self, outpoint: &OutPoint) -> Result<Amount, anyhow::Error>;
}

pub trait ApplyTx {
    fn apply_tx(&mut self, tx: &Transaction);
}

impl<C: ChainState + Debug> MempoolImpl<C> {
    fn verify_inputs_available(&self, tx: &Transaction) -> Result<(), TxValidationError> {
        tx.get_inputs()
            .iter()
            .map(TxInput::get_outpoint)
            .find(|outpoint| !self.outpoint_available(outpoint))
            .map_or_else(
                || Ok(()),
                |outpoint| {
                    Err(TxValidationError::OutPointNotFound {
                        outpoint: outpoint.clone(),
                        tx: tx.clone(),
                    })
                },
            )
    }

    // This function returns true even if the outpoint is marked as Spent because of another
    // transaction that has been accepted to the mempool
    fn outpoint_available(&self, outpoint: &OutPoint) -> bool {
        self.store.contains_outpoint(outpoint) || self.chain_state.contains_outpoint(outpoint)
    }

    fn is_replaceable(&self, tx: Arc<Transaction>) -> bool {
        tx.is_replaceable()
            || self.store.ancestors(&tx.into()).iter().any(|entry| entry.tx.is_replaceable())
    }

    fn validate_transaction(&self, tx: &Transaction) -> Result<(), TxValidationError> {
        if tx.get_inputs().is_empty() {
            return Err(TxValidationError::NoInputs);
        }

        if tx.get_outputs().is_empty() {
            return Err(TxValidationError::NoOutputs);
        }

        if tx.is_coinbase() {
            return Err(TxValidationError::LooseCoinbase);
        }

        // TODO consier a MAX_MONEY check reminiscent of bitcoin's
        // TODO consider rejecting non-standard transactions
        if has_duplicate_entry(tx.get_inputs()) {
            return Err(TxValidationError::DuplicateInputs);
        }

        if tx.encoded_size() > MAX_BLOCK_SIZE {
            return Err(TxValidationError::ExceedsMaxBlockSize);
        }

        if self.contains_transaction(&tx.get_id()) {
            return Err(TxValidationError::TransactionAlreadyInMempool);
        }

        tx.get_inputs()
            .iter()
            .filter_map(|input| self.store.find_conflicting_tx(input.get_outpoint()))
            .all(|tx| self.is_replaceable(tx))
            .then(|| ())
            .ok_or(TxValidationError::ConflictWithIrreplaceableTransaction)?;

        self.verify_inputs_available(tx)?;

        //
        // TODO Understand why we need  two size checks, one in CheckTransaction, and one in
        // TODO Do we want to differentiate between standard and nonstandard transaction
        // TODO do we really want lock time to be u32? I think it should be an enum
        // TODO check finality, both tx wise and output wise
        //

        Ok(())
    }
}

impl<C: ChainState + Debug> Mempool<C> for MempoolImpl<C> {
    // TODO need to discuss parameters
    fn create(chain_state: C) -> Self {
        Self {
            store: MempoolStore::new(),
            chain_state,
        }
    }
    // TODO what are the inputs for this?
    fn new_tip_set(&mut self) -> Result<(), MempoolError> {
        Err(MempoolError::GenericError)
    }
    //

    fn add_transaction(&mut self, tx: Transaction) -> Result<(), MempoolError> {
        if self.store.txs_by_fee.len() >= MEMPOOL_MAX_TXS {
            return Err(MempoolError::MempoolFull);
        }
        self.validate_transaction(&tx)?;
        // TODO check how Bitcoin Core updates the chain state
        self.chain_state.apply_tx(&tx);
        self.store.add_transaction(tx)?;
        Ok(())
    }

    fn get_all(&self) -> Vec<&Transaction> {
        self.store.txs_by_fee.iter().map(|mempool_tx| mempool_tx.as_ref()).collect()
    }

    fn contains_transaction(&self, tx_id: &Id<Transaction>) -> bool {
        self.store.txs_by_id.contains_key(&tx_id.get())
    }

    // TODO what to do if the transaction is not found?
    // For now I choose to do nothing, as is done in bitcoin
    fn drop_transaction(&mut self, tx_id: &Id<Transaction>) {
        if let Some(tx) = self.store.txs_by_id.remove(&tx_id.get()) {
            self.store.txs_by_fee.remove(&tx);
        }
    }
}

pub trait Mempool<C> {
    fn create(chain_state: C) -> Self;
    fn add_transaction(&mut self, tx: Transaction) -> Result<(), MempoolError>;
    //TODO maybe better to return some `TransactionIterator` or something
    fn get_all(&self) -> Vec<&Transaction>;
    fn contains_transaction(&self, tx: &Id<Transaction>) -> bool;
    fn drop_transaction(&mut self, tx: &Id<Transaction>);
    fn new_tip_set(&mut self) -> Result<(), MempoolError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OutPointState {
    Spent,
    Unspent,
}

#[derive(Debug, Clone)]
struct ChainStateMock {
    txs: HashMap<Id<Transaction>, Transaction>,
    outpoints: HashMap<OutPoint, OutPointState>,
}

impl ChainState for ChainStateMock {}

impl ApplyTx for ChainStateMock {
    fn apply_tx(&mut self, tx: &Transaction) {
        self.txs.insert(tx.get_id(), tx.clone());
        let spent_by_tx: HashSet<_> =
            tx.get_inputs().iter().map(|input| input.get_outpoint()).collect();
        let created_by_tx = tx.get_outputs().iter().enumerate().map(|(index, _)| {
            (
                OutPoint::new(tx.get_id(), index as u32),
                OutPointState::Unspent,
            )
        });
        self.outpoints
            .iter_mut()
            .filter(|(outpoint, _)| spent_by_tx.contains(outpoint))
            .for_each(|(_, status)| *status = OutPointState::Spent);
        self.outpoints.extend(created_by_tx);
    }
}

impl ChainStateView for ChainStateMock {
    fn contains_outpoint(&self, outpoint: &OutPoint) -> bool {
        matches!(self.outpoints.get(outpoint), Some(..))
    }

    fn get_outpoint_value(&self, outpoint: &OutPoint) -> Result<Amount, anyhow::Error> {
        self.txs
            .get(&outpoint.get_tx_id())
            .ok_or_else(|| anyhow::Error::msg("Tx not found"))
            .and_then(|tx| {
                tx.get_outputs()
                    .get(outpoint.get_output_index() as usize)
                    .ok_or_else(|| anyhow::Error::msg("Output not found at index"))
                    .map(|output| output.get_value())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::address::Address;
    use common::chain::config::MAINNET_ADDRESS_PREFIX;
    use common::chain::transaction::{Destination, TxInput, TxOutput};
    use rand::Rng;

    const DUMMY_WITNESS_MSG: &[u8] = b"dummy_witness_msg";

    impl<C: ChainStateView> MempoolImpl<C> {
        fn get_chain_state(&self) -> &C {
            &self.chain_state
        }
    }

    impl ChainStateMock {
        fn new() -> Self {
            let genesis_tx = create_genesis_tx();
            let outpoints = genesis_tx
                .get_outputs()
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    (
                        OutPoint::new(genesis_tx.get_id(), index as u32),
                        OutPointState::Unspent,
                    )
                })
                .collect();
            Self {
                txs: std::iter::once((genesis_tx.get_id(), genesis_tx)).collect(),
                outpoints,
            }
        }
    }

    struct TxGenerator {
        chain_state: ChainStateMock,
        num_inputs: usize,
        num_outputs: usize,
    }

    impl TxGenerator {
        fn new(chain_state: &ChainStateMock, num_inputs: usize, num_outputs: usize) -> Self {
            Self {
                chain_state: chain_state.clone(),
                num_inputs,
                num_outputs,
            }
        }

        fn generate_tx(mut self) -> anyhow::Result<Transaction> {
            let inputs = self.generate_tx_inputs();
            let outputs = self.generate_tx_outputs(&inputs)?;
            let locktime = 0;
            let flags = 0;
            Transaction::new(flags, inputs, outputs, locktime).map_err(Into::into)
        }

        fn generate_replaceable_tx(mut self) -> anyhow::Result<Transaction> {
            let inputs = self.generate_tx_inputs();
            let outputs = self.generate_tx_outputs(&inputs)?;
            let locktime = 0;
            let flags = 1;
            let tx = Transaction::new(flags, inputs, outputs, locktime)?;
            assert!(tx.is_replaceable());
            Ok(tx)
        }

        fn generate_tx_inputs(&mut self) -> Vec<TxInput> {
            std::iter::repeat(())
                .take(self.num_inputs)
                .filter_map(|_| self.generate_input().ok())
                .collect()
        }

        fn generate_tx_outputs(&self, inputs: &[TxInput]) -> anyhow::Result<Vec<TxOutput>> {
            if inputs.is_empty() {
                return Ok(vec![]);
            }
            let max_spend = inputs
                .iter()
                .map(|input| {
                    self.chain_state
                        .get_outpoint_value(input.get_outpoint())
                        .expect("outpoint not found")
                })
                .sum::<Option<_>>()
                .expect("Overflow in sum of input values");

            let mut left_to_spend = u128::from(max_spend);
            let mut outputs = Vec::new();

            let mut rng = rand::thread_rng();
            const MAX_OUTPUT_VALUE: u128 = 1_000;
            for _ in 0..self.num_outputs {
                let max_output_value = std::cmp::min(left_to_spend / 2, MAX_OUTPUT_VALUE);
                if max_output_value == 0 {
                    return Err(anyhow::Error::msg("No more funds to spend"));
                }
                let value = rng.gen_range(1..=max_output_value);
                outputs.push(TxOutput::new(Amount::from(value), Destination::PublicKey));
                left_to_spend -= value;
            }
            Ok(outputs)
        }

        fn generate_input(&mut self) -> anyhow::Result<TxInput> {
            self.random_unspent_outpoint().map(|outpoint| {
                self.chain_state.outpoints.remove(&outpoint);
                TxInput::new(
                    outpoint.get_tx_id(),
                    outpoint.get_output_index(),
                    DUMMY_WITNESS_MSG.to_vec(),
                )
            })
        }

        fn random_unspent_outpoint(&self) -> anyhow::Result<OutPoint> {
            let unspent_outpoints = self.unspent_outpoints();
            let num_outpoints = unspent_outpoints.len();
            (num_outpoints > 0)
                .then(|| {
                    let index = rand::thread_rng().gen_range(0..num_outpoints);
                    unspent_outpoints
                        .iter()
                        .nth(index)
                        .expect("Outpoint set should not be empty")
                        .clone()
                })
                .ok_or(anyhow::anyhow!("no outpoints left"))
        }

        fn unspent_outpoints(&self) -> HashSet<OutPoint> {
            self.chain_state
                .outpoints
                .iter()
                .filter_map(|(outpoint, state)| {
                    (*state == OutPointState::Unspent).then(|| outpoint)
                })
                .cloned()
                .collect()
        }
    }

    const TOTAL_SUPPLY: u128 = 10_000_000_000_000;

    fn create_genesis_tx() -> Transaction {
        let genesis_message = b"".to_vec();
        let genesis_mint_receiver = Address::new_with_hrp(MAINNET_ADDRESS_PREFIX, [])
            .expect("Failed to create genesis mint address");
        let input = TxInput::new(Id::new(&H256::zero()), 0, genesis_message);
        let output = TxOutput::new(
            Amount::new(TOTAL_SUPPLY),
            Destination::Address(genesis_mint_receiver),
        );
        Transaction::new(0, vec![input], vec![output], 0)
            .expect("Failed to create genesis coinbase transaction")
    }

    #[test]
    fn add_single_tx() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());

        let genesis_tx =
            mempool.get_chain_state().txs.values().next().expect("genesis tx not found");

        let input = TxInput::new(genesis_tx.get_id(), 0, DUMMY_WITNESS_MSG.to_vec());
        let outputs = spend_input(mempool.get_chain_state(), &input)?;

        let flags = 0;
        let inputs = vec![input];
        let locktime = 0;
        let tx = Transaction::new(flags, inputs, outputs, locktime)
            .map_err(|e| anyhow::anyhow!("failed to create transaction: {:?}", e))?;

        let tx_clone = tx.clone();
        let tx_id = tx.get_id();
        mempool.add_transaction(tx)?;
        assert!(mempool.contains_transaction(&tx_id));
        let all_txs = mempool.get_all();
        assert_eq!(all_txs, vec![&tx_clone]);
        mempool.drop_transaction(&tx_id);
        assert!(!mempool.contains_transaction(&tx_id));
        let all_txs = mempool.get_all();
        assert_eq!(all_txs, Vec::<&Transaction>::new());
        Ok(())
    }

    // The "fees" now a are calculated as sum of the outputs
    // This test creates transactions with a single input and a single output to check that the
    // mempool sorts txs by fee
    #[test]
    fn txs_sorted() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());
        let target_txs = 100;

        for _ in 0..target_txs {
            let num_inputs = 1;
            let num_outputs = 1;
            match TxGenerator::new(mempool.get_chain_state(), num_inputs, num_outputs).generate_tx()
            {
                Ok(tx) => {
                    mempool.add_transaction(tx.clone())?;
                }
                _ => break,
            }
        }

        let fees = mempool
            .get_all()
            .iter()
            .map(|tx| {
                tx.get_outputs().first().expect("TX should have exactly one output").get_value()
            })
            .collect::<Vec<_>>();
        let mut fees_sorted = fees.clone();
        fees_sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(fees, fees_sorted);
        Ok(())
    }

    #[test]
    fn tx_no_inputs() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());
        let num_inputs = 0;
        let num_outputs = 1;
        let tx = TxGenerator::new(mempool.get_chain_state(), num_inputs, num_outputs)
            .generate_tx()
            .expect("generate_tx failed");
        assert!(matches!(
            mempool.add_transaction(tx),
            Err(MempoolError::TxValidationError(TxValidationError::NoInputs))
        ));
        Ok(())
    }

    #[test]
    fn tx_no_outputs() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());
        let num_inputs = 1;
        let num_outputs = 0;
        let tx = TxGenerator::new(mempool.get_chain_state(), num_inputs, num_outputs)
            .generate_tx()
            .expect("generate_tx failed");
        assert!(matches!(
            mempool.add_transaction(tx),
            Err(MempoolError::TxValidationError(
                TxValidationError::NoOutputs
            ))
        ));
        Ok(())
    }

    fn spend_input(chain_state: &ChainStateMock, input: &TxInput) -> anyhow::Result<Vec<TxOutput>> {
        let input_value = chain_state.get_outpoint_value(input.get_outpoint())?;
        let output_value = (input_value / 2.into()).expect("failed to divide input");
        let output_pay = TxOutput::new(output_value, Destination::PublicKey);
        let output_change_amount = (input_value - output_value).expect("underflow");
        let output_change = TxOutput::new(output_change_amount, Destination::PublicKey);
        Ok(vec![output_pay, output_change])
    }

    #[test]
    fn tx_duplicate_inputs() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());

        let genesis_tx =
            mempool.get_chain_state().txs.values().next().expect("genesis tx not found");

        let input = TxInput::new(genesis_tx.get_id(), 0, DUMMY_WITNESS_MSG.to_vec());
        let outputs = spend_input(mempool.get_chain_state(), &input)?;
        let inputs = vec![input.clone(), input];
        let flags = 0;
        let locktime = 0;
        let tx = Transaction::new(flags, inputs, outputs, locktime)
            .map_err(|e| anyhow::anyhow!("failed to create transaction: {:?}", e))?;

        assert!(matches!(
            mempool.add_transaction(tx),
            Err(MempoolError::TxValidationError(
                TxValidationError::DuplicateInputs
            ))
        ));
        Ok(())
    }

    #[test]
    fn tx_already_in_mempool() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());

        let genesis_tx =
            mempool.get_chain_state().txs.values().next().expect("genesis tx not found");

        let input = TxInput::new(genesis_tx.get_id(), 0, DUMMY_WITNESS_MSG.to_vec());
        let outputs = spend_input(mempool.get_chain_state(), &input)?;

        let flags = 0;
        let inputs = vec![input];
        let locktime = 0;
        let tx = Transaction::new(flags, inputs, outputs, locktime)?;

        mempool.add_transaction(tx.clone())?;
        assert!(matches!(
            mempool.add_transaction(tx),
            Err(MempoolError::TxValidationError(
                TxValidationError::TransactionAlreadyInMempool
            ))
        ));
        Ok(())
    }

    pub fn coinbase_input() -> TxInput {
        TxInput::new(
            Id::new(&H256::zero()),
            OutPoint::COINBASE_OUTPOINT_INDEX,
            DUMMY_WITNESS_MSG.to_vec(),
        )
    }

    pub fn coinbase_output() -> TxOutput {
        const BLOCK_REWARD: u32 = 50;
        TxOutput::new(Amount::new(BLOCK_REWARD.into()), Destination::PublicKey)
    }

    pub fn coinbase_tx() -> anyhow::Result<Transaction> {
        const COINBASE_LOCKTIME: u32 = 100;

        let flags = 0;
        let inputs = vec![coinbase_input()];
        let outputs = vec![coinbase_output()];
        let locktime = COINBASE_LOCKTIME;
        Transaction::new(flags, inputs, outputs, locktime).map_err(anyhow::Error::from)
    }

    #[test]
    fn loose_coinbase() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());
        let coinbase_tx = coinbase_tx()?;

        assert!(matches!(
            mempool.add_transaction(coinbase_tx),
            Err(MempoolError::TxValidationError(
                TxValidationError::LooseCoinbase
            ))
        ));
        Ok(())
    }

    #[test]
    fn outpoint_not_found() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());

        let genesis_tx =
            mempool.get_chain_state().txs.values().next().expect("genesis tx not found");

        let good_input = TxInput::new(genesis_tx.get_id(), 0, DUMMY_WITNESS_MSG.to_vec());
        let outputs = spend_input(mempool.get_chain_state(), &good_input)?;
        let bad_outpoint_index = 1;
        let bad_input = TxInput::new(
            genesis_tx.get_id(),
            bad_outpoint_index,
            DUMMY_WITNESS_MSG.to_vec(),
        );

        let flags = 0;
        let inputs = vec![bad_input];
        let locktime = 0;
        let tx = Transaction::new(flags, inputs, outputs, locktime)?;

        assert!(matches!(
            mempool.add_transaction(tx),
            Err(MempoolError::TxValidationError(
                TxValidationError::OutPointNotFound { .. }
            ))
        ));

        Ok(())
    }

    #[test]
    fn tx_too_big() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());
        let num_inputs = 1;
        let num_outputs = 400_000;
        let tx = TxGenerator::new(mempool.get_chain_state(), num_inputs, num_outputs)
            .generate_tx()
            .expect("generate_tx failed");
        assert!(matches!(
            mempool.add_transaction(tx),
            Err(MempoolError::TxValidationError(
                TxValidationError::ExceedsMaxBlockSize
            ))
        ));
        Ok(())
    }

    #[test]
    fn tx_replace() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());
        let num_inputs = 1;
        let num_outputs = 1;
        let tx = TxGenerator::new(mempool.get_chain_state(), num_inputs, num_outputs)
            .generate_replaceable_tx()
            .expect("generate_replaceable_tx");
        mempool.add_transaction(tx)?;

        let tx = TxGenerator::new(mempool.get_chain_state(), num_inputs, num_outputs)
            .generate_tx()
            .expect("generate_tx_failed");

        mempool.add_transaction(tx)?;
        Ok(())
    }

    #[test]
    fn tx_replace_child() -> anyhow::Result<()> {
        let mut mempool = MempoolImpl::create(ChainStateMock::new());
        let num_inputs = 1;
        let num_outputs = 1;
        let tx = TxGenerator::new(mempool.get_chain_state(), num_inputs, num_outputs)
            .generate_replaceable_tx()
            .expect("generate_replaceable_tx");
        mempool.add_transaction(tx.clone())?;
        println!("first tx added successfully");

        let child_tx_input = TxInput::new(tx.get_id(), 0, DUMMY_WITNESS_MSG.to_vec());
        // We want to test that even though it doesn't signal replaceability directly, the child tx is replaceable because it's parent signalled replaceability
        // replaced
        let flags = 0;
        let locktime = 0;
        let outputs = spend_input(mempool.get_chain_state(), &child_tx_input)?;
        let inputs = vec![child_tx_input];
        let child_tx = Transaction::new(flags, inputs, outputs, locktime)?;
        mempool.add_transaction(child_tx)?;
        Ok(())
    }
}

fn has_duplicate_entry<T>(iter: T) -> bool
where
    T: IntoIterator,
    T::Item: Eq + std::hash::Hash,
{
    let mut uniq = HashSet::new();
    iter.into_iter().any(move |x| !uniq.insert(x))
}
