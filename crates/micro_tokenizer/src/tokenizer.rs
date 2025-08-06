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
        tokens.sort_by(|(a, _), (b, _)| {
            let a = expanded.get(a.clone()).expect("Token does not exist");
            let b = expanded.get(b.clone()).expect("Token does not exist");

            a.cmp(b)
        });

        // let mut cached_range = [Range::default(); 256];
        
        // TODO: remove the unsafe... i need to sleep now, so fuck it, will fix later
        let mut cached_range = unsafe { std::mem::zeroed::<[TokenRange; 256]>() };

        let mut offset = 0;
        for (i, range) in cached_range.iter_mut().enumerate() {
            let start = offset;

            while {
                if let Some(tok_range) = tokens.get(offset) {
                    let expanded = expanded.get(tok_range.0.clone()).expect("Token does not exist");

                    expanded[0] == i as u8
                } else {
                    false
                }
            } {
                offset += 1;
            }

            let end = offset;

            *range = start..end;
        }
        
        
        Self {
            vocab,
            expanded,
            cached_range,
            tokens
        }
        
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