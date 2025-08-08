pub mod load;
pub mod model;
pub mod nn;

use half::bf16;

#[test]
fn load_safetensors() {
    use memmap2::MmapOptions;
    use safetensors::SafeTensors;

    let file = std::fs::File::open(
        "/run/media/shark/datagrave/ai-modles/huggingface/llama-3.2-1B/model.safetensors",
    )
    .unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };

    let model = SafeTensors::deserialize(&buffer).unwrap();

    for layer in model.iter() {
        println!(
            "layer: {}: {:?} ({:?})",
            layer.0,
            layer.1.dtype(),
            layer.1.shape()
        );
    }

    let layer = model.tensor("model.norm.weight").unwrap();

    let data = bytemuck::cast_slice::<u8, bf16>(layer.data());

    let array = ndarray::ArrayView::from_shape(layer.shape(), data).unwrap();

    dbg!(array);

    let mut names = model.names();

    names.sort();

    for name in names {
        println!("name: {}", name);
    }
}
