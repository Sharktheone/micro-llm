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
    pub fn from_vocab(raw_vocab: &[Vec<u8>]) -> Self {
        let mut expanded = Vec::with_capacity(raw_vocab.iter().map(|v| v.len()).sum());
        let mut vocab = Vec::with_capacity(raw_vocab.len());
        let mut tokens = Vec::with_capacity(raw_vocab.len());
        
        for token in raw_vocab {
            let start = expanded.len();
            expanded.extend_from_slice(token);
            let end = expanded.len();
            vocab.push(start..end);
            tokens.push((start..end, tokens.len() as TokenId));
        }
        
        
        
        Self::from_raw(expanded, vocab, tokens)
    }

    pub fn from_unordered_vocab(raw_vocab: &[(Vec<u8>, TokenId)]) -> Self {
        let mut expanded = Vec::with_capacity(raw_vocab.iter().map(|v| v.0.len()).sum());
        let mut vocab = Vec::with_capacity(raw_vocab.len());
        let mut tokens = Vec::with_capacity(raw_vocab.len());

        for (token, id) in raw_vocab {
            let start = expanded.len();
            expanded.extend_from_slice(token);
            let end = expanded.len();
            
            let idx = *id as usize;
            
            if idx >= vocab.len() {
                vocab.resize(idx + 1, usize::MAX..usize::MAX);
            }
            
            vocab.insert(idx, start..end);
            
            
            tokens.push((start..end, *id));
        }
        
        vocab.iter().any(|r| r.start == usize::MAX && r.end == usize::MAX)
            .then(|| {
                panic!("Unordered vocab contains gaps, which is not supported by this tokenizer.");
            });

        Self::from_raw(expanded, vocab, tokens)
        
    }
    
    pub fn from_raw(mut expanded: Vec<u8>, mut vocab: Vec<TokenRange>, mut tokens: Vec<(TokenRange, TokenId)>) -> Self {
        
        todo!()
        
    }

    pub fn encode(&self, text: impl AsRef<[u8]>) -> Vec<TokenId> {
        Vec::new()
    }

    pub fn decode(&self, tokens: &[TokenId]) -> String {
        let bytes = self.decode_bytes(tokens);

        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn decode_bytes(&self, tokens: &[TokenId]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(tokens.len() * 6);

        for token in tokens {
            let Some(range) = self.vocab.get(*token as usize) else {
                continue;
            };

            let Some(voc) = self.expanded.get(range.clone()) else {
                continue;
            };


            bytes.extend_from_slice(voc);
        }


        bytes


    }
}