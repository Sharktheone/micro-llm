use std::cmp::Ordering;

pub type TokenId = u32;

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

        vocab
            .iter()
            .any(|r| r.start == u32::MAX && r.end == u32::MAX)
            .then(|| {
                panic!("Unordered vocab contains gaps, which is not supported by this tokenizer.");
            });

        Self::from_raw(expanded, vocab, tokens)
    }

    pub fn from_raw(
        expanded: Vec<u8>,
        vocab: Vec<Range>,
        mut tokens: Vec<(Range, TokenId)>,
    ) -> Self {
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
            tokens,
        }
    }

    pub fn encode(&self, b: impl AsRef<[u8]>) -> Vec<TokenId> {
        let b = b.as_ref();

        let mut buf = Vec::with_capacity(b.len() / 3);

        let mut i = 0;

        while i < b.len() {
            let (token, len) = self.search_max_token(&b[i..]);

            buf.push(token);

            i += len;
        }

        buf
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

    pub fn decode_token(&self, token: TokenId) -> Option<String> {
        let range = self.vocab.get(token as usize)?;

        let voc = self.expanded.get_range(*range);

        String::from_utf8(voc.to_vec()).ok()
    }

    fn search_max_token(&self, buf: &[u8]) -> (u32, usize) {
        let Some(pref) = buf.first().copied() else {
            return (u32::MAX, 0);
        };

        let range = self.cached_range[pref as usize];

        let tokens = self.tokens.get_range(range);

        if tokens.is_empty() {
            return (u32::MAX, 1);
        }

        let Some(mut search) = buf.get(1).cloned() else {
            let buf = self.expanded.get_range(tokens[0].0);
            if buf[0] == pref && buf.len() == 1 {
                return (tokens[0].1, 1);
            }

            //TODO: we should now search for the next few tokens
            return (u32::MAX, 1);
        };

        let mut left = 0;
        let mut right = tokens.len() as u32;
        let mut round = 1;
        let mut size = tokens.len() as u32;
        let mut cur_token = tokens[0].1;
        let mut cur_len = 1;

        let mut real_tok_idx = None;

        let ret = {
            #[inline(always)]
            |cur_token: u32, cur_len: usize, real_tok_idx: Option<u32>| {
                if let Some(idx) = real_tok_idx {
                    if let Some((range, tok)) = tokens.get(idx as usize) {
                        let expanded = self.expanded.get_range(*range);

                        if buf.len() > expanded.len() && &buf[..expanded.len()] == expanded {
                            return (*tok, expanded.len());
                        }
                    }
                }

                (cur_token, cur_len)
            }
        };

        while left < right {
            let mid = left + size / 2;

            let (range, tok) = tokens[mid as usize];

            let expanded = self.expanded.get_range(range);

            // if we don't have any here, this means that the token is less than our search
            let a = expanded.get(round).cloned();

            let Some(a) = a else {
                left = mid + 1;

                size = right - left;

                continue;
            };

            let cmp = a.cmp(&search);

            match cmp {
                Ordering::Equal => {
                    let mut size = mid - left;

                    let mut left_right = mid;

                    //search for the new left bound
                    while left < left_right {
                        let mid = left + size / 2;

                        let (a, _) = tokens[mid as usize];

                        let expanded = self.expanded.get_range(a);

                        let Some(a) = expanded.get(round).cloned() else {
                            left = mid + 1;
                            size = left_right - left;
                            continue;
                        };

                        // `a` can only be less or equal search
                        if a < search {
                            left = mid + 1;
                            size = left_right - left;
                            continue;
                        }

                        // we checked that `a` isn't less  than search, so it must be equal

                        let (b, _) = tokens[mid as usize - 1];

                        let expanded = self.expanded.get_range(b);
                        let b = expanded.get(round).cloned();

                        // if b is none or, it means we have found our location
                        let Some(b) = b else {
                            left = mid;
                            break;
                        };

                        // if b is less than search, we have found our location
                        if b < search {
                            left = mid;
                            break;
                        }

                        // b is less than search, so we need to move left

                        left_right = mid;
                        size = left_right - left;
                    }

                    // search for the new right bound
                    let mut size = right - mid;
                    let mut right_left = mid;
                    while right_left < right {
                        let mid = right_left + size / 2;

                        let (a, _) = tokens[mid as usize - 1];

                        let expanded = self.expanded.get_range(a);

                        let Some(a) = expanded.get(round).cloned() else {
                            // `a` is none, we are too far
                            right = mid;
                            size = right - right_left;
                            continue;
                        };

                        // `a` can only be greater or equal search
                        if a > search {
                            // `a` is greater than search, we are too far
                            right = mid;
                            size = right - right_left;
                            continue;
                        }

                        // we checked that `a` isn't greater than search, so it must be equal

                        let (b, _) = tokens[mid as usize];

                        let expanded = self.expanded.get_range(b);
                        let b = expanded.get(round).cloned();

                        // if b is none, it means we have found our location
                        let Some(b) = b else {
                            right = mid;
                            break;
                        };

                        // if b is greater than search, we have found our location
                        if b > search {
                            right = mid;
                            break;
                        }

                        // b is equal to search, so we need to move right

                        right_left = mid + 1;
                        size = right - right_left;
                    }

                    round += 1;
                    if round == range.len() as usize {
                        cur_token = tok;
                        cur_len = round;
                        real_tok_idx = None;
                    } else {
                        let (range, tok) = tokens[left as usize];

                        if range.len() as usize == round {
                            cur_token = tok;
                            cur_len = round;
                            real_tok_idx = None;
                        } else {
                            real_tok_idx = Some(mid)
                        }
                    }

                    let Some(s) = buf.get(round).cloned() else {
                        return ret(cur_token, cur_len, real_tok_idx);
                    };

                    search = s;
                }
                Ordering::Less => left = mid + 1,
                Ordering::Greater => right = mid,
            }

            size = right - left;
        }

        ret(cur_token, cur_len, real_tok_idx)
    }

    pub fn bytes_allocated(&self) -> usize {
        self.expanded.capacity()
            + self.vocab.capacity() * size_of::<Range>()
            + self.tokens.capacity() * size_of::<(Range, TokenId)>()
            + size_of::<Self>()
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
