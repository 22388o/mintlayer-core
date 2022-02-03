use common::address::Address;
use common::primitives::BlockHeight;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Config {
    pub(crate) max_num_of_validator_slots: u8,
    pub(crate) min_num_of_validators: u8,
    // a period of time, measured in block height
    pub(crate) epoch_length: BlockHeight,
}
