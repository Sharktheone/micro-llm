use std::fs;
use crate::tokenizer::Tokenizer;

pub fn load_hf_tokenizer(path: &str) -> anyhow::Result<Tokenizer> {
    let vocab_file = fs::read_to_string(path)?;
    let vocab_contents = serde_json::from_str::<serde_json::Value>(&vocab_file)?;


    let tokens = vocab_contents.get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("No vocab_file found in tokenizer.json"))?;

    let mut token_ids = Vec::with_capacity(tokens.len());

    for (token, id) in tokens.iter() {
        if let Some(id) = id.as_u64() {
            token_ids.push((token.as_bytes().to_vec(), id as u32));
        }
    }

    let tokenizer = Tokenizer::from_unordered_vocab(&token_ids);

    Ok(tokenizer)

}