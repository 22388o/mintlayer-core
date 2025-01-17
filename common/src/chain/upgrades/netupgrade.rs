#![allow(clippy::upper_case_acronyms, clippy::needless_doctest_main)]

use crate::primitives::BlockHeight;

#[derive(Debug, Clone)]
pub struct NetUpgrades<T>(Vec<(BlockHeight, T)>);

impl<T: Default> Default for NetUpgrades<T> {
    fn default() -> Self {
        Self(vec![(BlockHeight::zero(), T::default())])
    }
}

pub trait Activate {
    fn is_activated(&self, height: BlockHeight, net_upgrades: &NetUpgrades<Self>) -> bool
    where
        Self: Sized + Ord + Copy,
    {
        if let Ok(idx) = net_upgrades.0.binary_search_by(|&(_, to_match)| to_match.cmp(self)) {
            return height >= net_upgrades.0[idx].0;
        }
        false
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub enum UpgradeVersion {
    Genesis = 0,
    PoW,
    PoS,
    DSA,
}

impl Activate for UpgradeVersion {}

impl Default for UpgradeVersion {
    fn default() -> Self {
        Self::Genesis
    }
}

impl<T: Default + Ord + Copy> NetUpgrades<T> {
    #[allow(dead_code)]
    pub(crate) fn initialize(upgrades: Vec<(BlockHeight, T)>) -> Self {
        let mut upgrades = upgrades;
        upgrades.sort_unstable();

        if let Some(&(height, _)) = upgrades.first() {
            return if height == BlockHeight::zero() {
                Self(upgrades)
            } else {
                let mut default = Self::default();
                default.0.append(&mut upgrades);
                default
            };
        }

        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get_version(&self, height: BlockHeight) -> T {
        match self.0.iter().rfind(|&&(elem_height, _)| elem_height <= height) {
            None => T::default(),
            Some(&(_, version)) => version,
        }
    }

    pub fn height_range(&self, version: T) -> Option<(BlockHeight, BlockHeight)> {
        let res = self
            .0
            .iter()
            .enumerate()
            .find(|&(_, &(_, elem_version))| elem_version == version);

        res.map(|(idx, &(start_h, _))| {
            (
                start_h,
                if idx == (self.0.len() - 1) {
                    BlockHeight::max()
                } else {
                    BlockHeight::new(self.0[idx + 1].0.inner() - 1)
                },
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::chain::upgrades::netupgrade::NetUpgrades;
    use crate::chain::Activate;
    use crate::primitives::BlockHeight;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
    pub enum MockVersion {
        Zero,
        One,
        Two,
        Three,
        Four,
        Five,
    }

    impl Default for MockVersion {
        fn default() -> Self {
            Self::Zero
        }
    }

    impl Activate for MockVersion {}

    fn mock_netupgrades() -> (NetUpgrades<MockVersion>, BlockHeight, BlockHeight) {
        let mut upgrades = vec![];
        let two_height = BlockHeight::new(3500);
        let three_height = BlockHeight::new(80000);

        upgrades.push((three_height, MockVersion::Three));

        upgrades.push((BlockHeight::one(), MockVersion::One));

        upgrades.push((two_height, MockVersion::Two));

        (NetUpgrades::initialize(upgrades), two_height, three_height)
    }

    #[test]
    fn check_is_activated() {
        let (upgrades, two_height, three_height) = mock_netupgrades();

        assert!(MockVersion::Two.is_activated(two_height, &upgrades));
        assert!(MockVersion::Two.is_activated(three_height, &upgrades));
        assert!(!MockVersion::Two.is_activated(BlockHeight::one(), &upgrades));
        assert!(!MockVersion::Two.is_activated(BlockHeight::new(two_height.inner() - 1), &upgrades));

        assert!(!MockVersion::Three.is_activated(two_height, &upgrades));
        assert!(!MockVersion::Three.is_activated(
            two_height.checked_add(10).expect("should be fine"),
            &upgrades
        ));
        assert!(MockVersion::Three.is_activated(three_height, &upgrades));
        assert!(MockVersion::Three.is_activated(BlockHeight::max(), &upgrades));
    }

    #[test]
    fn check_upgrade_version_from_height() {
        let (upgrades, two_height, three_height) = mock_netupgrades();

        let check = |v: MockVersion, h: BlockHeight| {
            assert_eq!(v, upgrades.get_version(h));
        };

        check(MockVersion::Zero, BlockHeight::zero());
        check(MockVersion::One, BlockHeight::one());
        check(MockVersion::One, BlockHeight::new(26));
        check(MockVersion::One, BlockHeight::new(two_height.inner() - 1));
        check(MockVersion::Two, two_height);
        check(
            MockVersion::Two,
            two_height.checked_add(1).expect("should be fine"),
        );
        check(MockVersion::Two, BlockHeight::new(three_height.inner() - 1));
        check(MockVersion::Three, three_height);
        check(
            MockVersion::Three,
            three_height.checked_add(100).expect("should be fine"),
        );
        check(
            MockVersion::Three,
            three_height.checked_add(2022).expect("should be fine"),
        );
        check(
            MockVersion::Three,
            three_height.checked_add(3000).expect("should be fine"),
        );
    }

    #[test]
    fn check_upgrade_versions() {
        assert_eq!(0u8, MockVersion::Zero as u8);
        assert_eq!(1u8, MockVersion::One as u8);
        assert_eq!(2u8, MockVersion::Two as u8);
        assert_eq!(3u8, MockVersion::Three as u8);
        assert_eq!(4u8, MockVersion::Four as u8);
        assert_eq!(5u8, MockVersion::Five as u8);
    }

    #[test]
    fn check_upgrade_height_range() {
        let (upgrades, two_height, three_height) = mock_netupgrades();

        let check = |vers_type: MockVersion, height: BlockHeight, end_range: BlockHeight| {
            let res = upgrades.height_range(vers_type);

            assert_eq!(Some((height, end_range)), res);
        };

        check(MockVersion::Zero, BlockHeight::zero(), BlockHeight::zero());
        check(
            MockVersion::One,
            BlockHeight::one(),
            BlockHeight::new(two_height.inner() - 1),
        );
        check(
            MockVersion::Two,
            two_height,
            BlockHeight::new(three_height.inner() - 1),
        );
        check(MockVersion::Three, three_height, BlockHeight::max());
    }
}
