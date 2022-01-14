//! Set of functions providing an early exit from a function based on a boolean condition.
//!
//! This is a substitute for Parity's `ensure!` macro, without using macros:
//!
//! * `ensure!(cond, err);` becomes
//!   * [`ensure`]`(cond, err)?;` for simple errors
//!   * [`ensure_fn`]`(cond, || err)?;` for expensive errors
//! * `ensure!(cond);` (hypothetical variant giving an `Option`) becomes
//!   * [`ensure_some`]`(cond)?;`

/// Map `true` to `Ok(())` and `false` to `Err(error)`.
///
/// ```
/// # use common::util::ensure::*;
/// # #[derive(PartialEq, Eq, Debug)]
/// enum DivError { ByZero, Remainder };
///
/// fn integral_div(x: u32, y: u32) -> Result<u32, DivError> {
///     ensure(y != 0, DivError::ByZero)?;
///     ensure(x % y == 0, DivError::Remainder)?;
///     Ok(x / y)
/// }
///
/// assert_eq!(integral_div(12, 4), Ok(3));
/// assert_eq!(integral_div(11, 4), Err(DivError::Remainder));
/// assert_eq!(integral_div(11, 0), Err(DivError::ByZero));
/// ```
#[must_use]
#[inline]
pub fn ensure<E>(condition: bool, error: E) -> Result<(), E> {
    ensure_some(condition).ok_or(error)
}

/// Map `true` to `Ok(())` and `false` to `Err(error_fn())`.
///
/// Same as [`ensure`] with error calculated lazily.
#[must_use]
#[inline]
pub fn ensure_fn<E>(condition: bool, error_fn: impl FnOnce() -> E) -> Result<(), E> {
    ensure_some(condition).ok_or_else(error_fn)
}

/// Map `true` to `Some(())` and `false` to `None`.
///
/// ```
/// # use common::util::ensure::*;
/// fn safe_div(x: u32, y: u32) -> Option<u32> {
///     ensure_some(y != 0)?;
///     Some(x / y)
/// }
///
/// assert_eq!(safe_div(9, 3), Some(3));
/// assert_eq!(safe_div(9, 0), None);
/// ```
#[must_use]
#[inline]
pub fn ensure_some(condition: bool) -> Option<()> {
    condition.then(|| ())
}
