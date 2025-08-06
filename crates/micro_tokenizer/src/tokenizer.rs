
type TokenId = u32;

pub struct Tokenizer {
    expanded: Vec<u8>,
    vocab: Vec<Range>,
    tokens: Vec<(Range, TokenId)>,
    cached_range: [Range; 256],
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
            vocab.push(Range::new(start, end));
            tokens.push((Range::new(start, end), tokens.len() as TokenId));
        }
        
        
        
        Self::from_raw(expanded, vocab, tokens)
    }

    pub fn from_unordered_vocab(raw_vocab: &[(Vec<u8>, TokenId)]) -> Self {
        let mut expanded = Vec::with_capacity(raw_vocab.iter().map(|v| v.0.len()).sum());
        let mut vocab = vec![Range::new(usize::MAX, usize::MAX); raw_vocab.len()];
        let mut tokens = Vec::with_capacity(raw_vocab.len());

        for (token, id) in raw_vocab {
            let start = expanded.len();
            expanded.extend_from_slice(token);
            let end = expanded.len();
            
            let idx = *id as usize;
            
            vocab[idx] = Range::new(start, end);
            
            tokens.push((Range::new(start, end), *id));
        }
        
        vocab.iter().any(|r| r.start == u32::MAX && r.end == u32::MAX)
            .then(|| {
                panic!("Unordered vocab contains gaps, which is not supported by this tokenizer.");
            });

        Self::from_raw(expanded, vocab, tokens)
        
    }
    
    pub fn from_raw(expanded: Vec<u8>, vocab: Vec<Range>, mut tokens: Vec<(Range, TokenId)>) -> Self {
        tokens.sort_by(|(a, _), (b, _)| {
            let a = expanded.get_range(*a);
            let b = expanded.get_range(*b);

            a.cmp(b)
        });

        let mut cached_range = [Range::default(); 256];

        let mut offset = 0;
        for (i, range) in cached_range.iter_mut().enumerate() {
            let start = offset;

            while {
                if let Some(tok_range) = tokens.get(offset) {
                    let expanded = expanded.get_range(tok_range.0);

                    expanded[0] == i as u8
                } else {
                    false
                }
            } {
                offset += 1;
            }

            let end = offset;

            *range = Range::new(start, end);
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

            let voc = self.expanded.get_range(*range);


            bytes.extend_from_slice(voc);
        }


        bytes
    }
}



#[derive(Debug, Clone, Copy, Default)]
pub struct Range {
    start: u32,
    end: u32,
}

impl Range {
    fn new(start: usize, end: usize) -> Self {
        Self {
            start: start as u32,
            end: end as u32,
        }
    }

    pub fn len(&self) -> u32 {
        self.end - self.start
    }
}


pub(crate) trait VecExt<T> {
    fn get_range(&self, range: Range) -> &[T];
}

impl<T> VecExt<T> for Vec<T> {
    fn get_range(&self, range: Range) -> &[T] {
        let start = range.start as usize;
        let end = range.end as usize;

        &self[start..end]
    }
}
