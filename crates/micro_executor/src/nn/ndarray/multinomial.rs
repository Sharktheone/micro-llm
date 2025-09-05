use ndarray::Array1;
use ndarray_rand::rand;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, WeightedIndex};
use num_traits::Float;

pub fn multinomial<T: Float + Default + SampleUniform + for<'a> std::ops::AddAssign<&'a T>>(
    items: Array1<T>,
    num: usize,
) -> Array1<usize> {
    let dist = WeightedIndex::new(items);

    let mut rng = rand::thread_rng();

    dist.unwrap().sample_iter(&mut rng).take(num).collect()
}
