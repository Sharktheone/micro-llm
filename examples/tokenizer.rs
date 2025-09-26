use micro_tokenizer::hf::load_hf_tokenizer;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 || (args.len() == 3 && args[2] != "--file") {
        eprintln!("Usage: {} <tokenizer> <text>", args[0]);
        eprintln!("Usage: {} <tokenizer> --file <textfile>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    let hf = load_hf_tokenizer(&path).unwrap();

    let text = if args.len() == 3 {
        std::fs::read_to_string(&args[2]).unwrap()
    } else {
        args[2].clone()
    };

    let tokens = hf.encode(&text);

    println!("Tokens: {:?}", tokens);
}