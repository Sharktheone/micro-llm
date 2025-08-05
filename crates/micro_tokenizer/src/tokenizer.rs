use std::ops::Range;

type TokenRange = Range<usize>;
type TokenId = u32;

pub struct Tokenizer {
    expanded: Vec<u8>,
    vocab: Vec<TokenRange>,
    tokens: Vec<(TokenRange, TokenId)>,
    cached_range: [TokenRange; 256],
}

impl Tokenizer {
    pub fn from_vocab(vocab: &[Vec<u8>]) -> Self {
        todo!()
        
    }
    
    pub fn from_unordered_vocab(vocab: &[(Vec<u8>, TokenId)]) -> Self {
        todo!()
    }
    
    pub fn encode(&self, text: impl AsRef<[u8]>) -> Vec<TokenId> {
        Vec::new()
    }
    
    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        Vec::new()
    }
}