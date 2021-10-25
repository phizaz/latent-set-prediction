from transformers import RobertaTokenizerFast


def load_roberta_tokenizer(name='clevr'):
    tokenizer = RobertaTokenizerFast(
        f'./tokenizers/{name}-vocab.json',
        f'./tokenizers/{name}-merges.txt',
        sep_token='<sep>',
        cls_token='<cls>',
    )
    return tokenizer
