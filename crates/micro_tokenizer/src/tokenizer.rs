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