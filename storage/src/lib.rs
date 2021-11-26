use std::collections::BTreeMap;

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum DBError {
    Unknown,
}

pub trait DataType: AsRef<[u8]> + Clone {
}

impl DataType for Vec<u8> {
}

impl DataType for &[u8] {
}

/// number of unique databases within the database as a type
pub type DBIndexCountT = generic_array::typenum::U4;

pub trait Storage<I: ?Sized> {
    /// Returns true if a single key can have multiple values (non-unique keys)
    /// Notice that this doesn't take `self` because it's an interface definition, independent of the db library
    fn duplicates_allowed(db_index: I) -> bool;

    /// For a database with unique key/value, this sets the key vs value and overwrites an existing one
    /// If this were used with a databse allowing duplicates, the kv pair will become (k, [v])
    fn set<K: DataType, V: DataType>(
        &mut self,
        db_index: I,
        key: K,
        val: V,
    ) -> Result<(), DBError>;

    /// For a database with unique key/value, this gets the value stored with the given key
    /// If this were used with a databse not allowing duplicates, the result will provide only one value in the list of duplicates
    fn get<K: DataType>(
        &mut self,
        db_index: I,
        key: K,
        offset: usize,
        size: Option<usize>,
    ) -> Result<Option<Vec<u8>>, DBError>;

    /// For a database with non-unique key/values, this gets all the values stored with the given key
    /// If this were used with a databse not allowing duplicates, the result will provide only the unique value available in list
    fn get_multiple<K: DataType>(
        &mut self,
        db_index: I,
        key: K,
    ) -> Result<Vec<Vec<u8>>, DBError>;

    /// For a database that allows non-unique key/values, this returns everything in the database
    /// If this were used with a databse not allowing duplicates, the result will provide only the unique value available in list
    fn get_all(
        &mut self,
        db_index: I,
    ) -> Result<BTreeMap<Vec<u8>, Vec<Vec<u8>>>, DBError>;

    /// For a database with unique key/value, this gets all keys and their corresponding unique values
    /// If this were used with a database with non-unique values, only a single arbitrary value will be provided from the list of available values
    fn get_all_unique(
        &mut self,
        db_index: I,
    ) -> Result<BTreeMap<Vec<u8>, Vec<u8>>, DBError>;

    /// Returns true if the key exists in the database; false otherwise
    fn exists<K: DataType>(&mut self, db_index: I, key: K) -> Result<bool, DBError>;

    /// For a non-unique database, this appends a value to the available kv pairs
    /// If used with unique keys, an overwrite happens
    fn append<K: DataType, V: DataType>(&mut self, db_index: I, key: K, val: V) -> Result<(), DBError>;

    /// For a non-unique database, a key is erased with the respective value is erased
    /// For a unique database, the key is only erased if the key/value match
    fn erase_one<K: DataType, V: DataType>(
        &mut self,
        db_index: I,
        key: K,
        val: V,
    ) -> Result<(), DBError>;

    /// The key and all possible values are erased
    fn erase<K: DataType>(&mut self, db_index: I, key: K) -> Result<(), DBError>;

    /// All database content for all indexes are cleared
    fn clear_all(&mut self) -> Result<(), DBError>;

    /// All database content are cleared for the given index
    fn clear_db(&mut self, db_index: I) -> Result<(), DBError>;

    /// Begin an atomic, consistent and isolated transaction in the database
    /// this can be done once at a time, until reverted or committed
    fn begin_transaction(&mut self, approximate_data_size: usize) -> Result<(), DBError>;

    /// Revert all operations in the transaction that was started
    fn revert_transaction(&mut self) -> Result<(), DBError>;

    /// Commit the operations in the transaction to the persistent DB
    fn commit_transaction(&mut self) -> Result<(), DBError>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
