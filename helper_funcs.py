# Functions that ended up being used in multiple locations

def get_words_from_file(filename):
    with open(filename, 'r') as namesfile:
        words = namesfile.read()
    return words

def get_letters_tables(words):
    all_chars = sorted(set(''.join(words)))
    # if '.' not in all_chars:
    #     all_chars.insert(0, '.')
    # else:
    #     all_chars.insert(0, '0')
    letters_table = {s:i for i, s in enumerate(all_chars)}
    reverse_letters_table = {i:s for i, s in enumerate(all_chars)}
    return letters_table, reverse_letters_table