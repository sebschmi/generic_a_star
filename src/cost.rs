use std::{
    fmt::{Debug, Display},
    hash::Hash,
    iter::Sum,
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
    str::FromStr,
};

use num_traits::{Bounded, CheckedAdd, CheckedSub, SaturatingAdd, SaturatingSub, Zero};

/// The cost of an A* node.
pub trait AStarCost:
    From<u8>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + SaturatingAdd
    + SaturatingSub
    + CheckedAdd
    + CheckedSub
    + AddAssign
    + SubAssign
    + Sum
    + FromStr
    + Zero
    + Bounded
    + Display
    + Debug
    + Ord
    + Eq
    + Hash
    + Copy
{
    type CostType;

    fn as_f64(&self) -> f64;

    fn as_u64(&self) -> u64;

    fn as_primitive(&self) -> Self::CostType;

    fn from_primitive(value: Self::CostType) -> Self;

    fn as_usize(&self) -> usize;

    fn from_usize(value: usize) -> Self;
}

/// Panics with an overflow message for the given cost operation.
///
/// Kept out of line and marked cold, so that the inlined arithmetic operators stay small.
#[cold]
#[inline(never)]
#[track_caller]
fn cost_overflow_panic<Cost: Display>(operation: &str, operator: &str, lhs: Cost, rhs: Cost) -> ! {
    panic!("Overflow when {operation} costs {lhs} {operator} {rhs}")
}

macro_rules! primitive_cost {
    ($name:ident, $primitive:ident) => {
        #[doc = concat!("The cost of an A* node.\n\nThis cost type uses [`", stringify!($primitive), "`] as the internal representation of cost.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $name($primitive);

        impl $crate::cost::AStarCost for $name {
            type CostType = $primitive;

            #[inline]
            fn as_f64(&self) -> f64 {
                self.0 as f64
            }

            #[inline]
            fn as_u64(&self)->u64{
                self.0 as u64
            }

            #[inline]
            fn as_primitive(&self) -> Self::CostType {
                self.0
            }

            #[inline]
            fn from_primitive(value: Self::CostType) -> Self {
                Self(value)
            }

            #[inline]
            fn as_usize(&self) -> usize {
                self.as_primitive().try_into().unwrap()
            }

            #[inline]
            fn from_usize(value: usize) -> Self {
                Self::from_primitive(value.try_into().unwrap())
            }
        }

        impl From<$primitive> for $name {
            #[inline]
            fn from(value: $primitive) -> Self {
                Self(value)
            }
        }

        impl From<u8> for $name {
            #[inline]
            fn from(value: u8) -> Self {
                Self(value.into())
            }
        }

        impl std::ops::Add for $name {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                match self.0.checked_add(rhs.0) {
                    Some(value) => Self(value),
                    None => cost_overflow_panic("adding", "+", self, rhs),
                }
            }
        }

        impl std::ops::Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                match self.0.checked_sub(rhs.0) {
                    Some(value) => Self(value),
                    None => cost_overflow_panic("subtracting", "-", self, rhs),
                }
            }
        }

        impl std::ops::Mul for $name {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                match self.0.checked_mul(rhs.0) {
                    Some(value) => Self(value),
                    None => cost_overflow_panic("multiplying", "*", self, rhs),
                }
            }
        }

        impl num_traits::SaturatingAdd for $name {
            #[inline]
            fn saturating_add(&self, rhs: &Self) -> Self {
                Self(self.0.saturating_add(rhs.0))
            }
        }

        impl num_traits::SaturatingSub for $name {
            #[inline]
            fn saturating_sub(&self, rhs: &Self) -> Self {
                Self(self.0.saturating_sub(rhs.0))
            }
        }

        impl num_traits::CheckedAdd for $name {
            #[inline]
            fn checked_add(&self, rhs: &Self) -> Option<Self> {
                self.0.checked_add(rhs.0).map($name)
            }
        }

        impl num_traits::CheckedSub for $name {
            #[inline]
            fn checked_sub(&self, rhs: &Self) -> Option<Self> {
                self.0.checked_sub(rhs.0).map($name)
            }
        }

        impl std::ops::AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl std::ops::SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl std::iter::Sum for $name {
            #[inline]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::zero(), |acc, x| acc + x)
            }
        }

        impl std::fmt::Display for $name {
            #[inline]
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.0, f)
            }
        }

        impl std::str::FromStr for $name {
            type Err = <$primitive as std::str::FromStr>::Err;

            #[inline]
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                $primitive::from_str(s).map(Self)
            }
        }

        impl num_traits::Zero for $name {
            #[inline]
            fn zero() -> Self {
                $primitive::zero().into()
            }

            #[inline]
            fn is_zero(&self) -> bool {
                self.0.is_zero()
            }
        }

        impl num_traits::Bounded for $name {
            #[inline]
            fn min_value() -> Self {
                <$primitive as num_traits::Bounded>::min_value().into()
            }

            #[inline]
            fn max_value() -> Self {
                <$primitive as num_traits::Bounded>::max_value().into()
            }
        }
    };
}

primitive_cost!(U16Cost, u16);
primitive_cost!(I16Cost, i16);
primitive_cost!(U32Cost, u32);
primitive_cost!(I32Cost, i32);
primitive_cost!(U64Cost, u64);
primitive_cost!(I64Cost, i64);

/// A pair of cost types where the first cost is prioritised over the second cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrderedPairCost<A, B>(pub A, pub B);

impl<A: AStarCost, B: AStarCost> AStarCost for OrderedPairCost<A, B> {
    type CostType = A::CostType;

    #[inline]
    fn as_f64(&self) -> f64 {
        self.0.as_f64()
    }

    #[inline]
    fn as_u64(&self) -> u64 {
        self.0.as_u64()
    }

    #[inline]
    fn as_primitive(&self) -> Self::CostType {
        self.0.as_primitive()
    }

    #[inline]
    fn from_primitive(value: Self::CostType) -> Self {
        Self(A::from_primitive(value), B::zero())
    }

    #[inline]
    fn as_usize(&self) -> usize {
        self.0.as_usize()
    }

    #[inline]
    fn from_usize(value: usize) -> Self {
        Self(A::from_usize(value), B::zero())
    }
}

impl<A: Display, B: Display> Display for OrderedPairCost<A, B> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

impl<A: Zero, B: Zero> Zero for OrderedPairCost<A, B> {
    #[inline]
    fn zero() -> Self {
        Self(A::zero(), B::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

impl<A: Bounded, B: Bounded> Bounded for OrderedPairCost<A, B> {
    #[inline]
    fn min_value() -> Self {
        Self(A::min_value(), B::min_value())
    }

    #[inline]
    fn max_value() -> Self {
        Self(A::max_value(), B::max_value())
    }
}

impl<A: From<u8>, B: From<u8>> From<u8> for OrderedPairCost<A, B> {
    #[inline]
    fn from(value: u8) -> Self {
        Self(A::from(value), B::from(0))
    }
}

impl<A: Add<Output = A>, B: Add<Output = B>> Add for OrderedPairCost<A, B> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl<A: Sub<Output = A>, B: Sub<Output = B>> Sub for OrderedPairCost<A, B> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl<A: Mul<Output = A>, B: Mul<Output = B>> Mul for OrderedPairCost<A, B> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0, self.1 * rhs.1)
    }
}

impl<A: FromStr, B: FromStr> FromStr for OrderedPairCost<A, B> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let Some(s) = s.strip_prefix('(') else {
            return Err(());
        };
        let Some(s) = s.strip_suffix(')') else {
            return Err(());
        };
        let mut parts = s.splitn(2, ',');
        let s = parts.next().ok_or(())?.trim();
        let a = s.parse::<A>().map_err(|_| ())?;
        let s = parts.next().ok_or(())?.trim();
        let b = s.parse::<B>().map_err(|_| ())?;
        Ok(Self(a, b))
    }
}

impl<A: AddAssign, B: AddAssign> AddAssign for OrderedPairCost<A, B> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl<A: SubAssign, B: SubAssign> SubAssign for OrderedPairCost<A, B> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}

impl<A: CheckedAdd, B: CheckedAdd> CheckedAdd for OrderedPairCost<A, B> {
    #[inline]
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(Self(
            self.0.checked_add(&rhs.0)?,
            self.1.checked_add(&rhs.1)?,
        ))
    }
}

impl<A: CheckedSub, B: CheckedSub> CheckedSub for OrderedPairCost<A, B> {
    #[inline]
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(Self(
            self.0.checked_sub(&rhs.0)?,
            self.1.checked_sub(&rhs.1)?,
        ))
    }
}

impl<A: SaturatingAdd, B: SaturatingAdd> SaturatingAdd for OrderedPairCost<A, B> {
    #[inline]
    fn saturating_add(&self, rhs: &Self) -> Self {
        Self(self.0.saturating_add(&rhs.0), self.1.saturating_add(&rhs.1))
    }
}

impl<A: SaturatingSub, B: SaturatingSub> SaturatingSub for OrderedPairCost<A, B> {
    #[inline]
    fn saturating_sub(&self, rhs: &Self) -> Self {
        Self(self.0.saturating_sub(&rhs.0), self.1.saturating_sub(&rhs.1))
    }
}

impl<A: Sum + Zero, B: Sum + Zero> Sum for OrderedPairCost<A, B> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests {
    use super::{AStarCost, U32Cost};

    #[test]
    fn arithmetic() {
        let a = U32Cost::from(5u8);
        let b = U32Cost::from(3u8);
        assert_eq!(a + b, U32Cost::from(8u8));
        assert_eq!(a - b, U32Cost::from(2u8));
        assert_eq!(a * b, U32Cost::from(15u8));
        assert_eq!(U32Cost::from_usize(7).as_usize(), 7);
    }

    #[test]
    #[should_panic(expected = "Overflow when adding costs 4294967295 + 1")]
    fn add_overflow() {
        let _ = U32Cost::from(u32::MAX) + U32Cost::from(1u8);
    }

    #[test]
    #[should_panic(expected = "Overflow when subtracting costs 0 - 1")]
    fn sub_overflow() {
        let _ = U32Cost::from(0u8) - U32Cost::from(1u8);
    }

    #[test]
    #[should_panic(expected = "Overflow when multiplying costs 4294967295 * 2")]
    fn mul_overflow() {
        let _ = U32Cost::from(u32::MAX) * U32Cost::from(2u8);
    }
}
