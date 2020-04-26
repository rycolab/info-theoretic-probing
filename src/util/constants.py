import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LANGUAGE_CODES = {
    'english': 'en',
    'czech': 'cs',
    'basque': 'eu',
    'finnish': 'fi',
    'turkish': 'tr',
    'tamil': 'ta',
    'korean': 'ko',
    'marathi': 'mr',
    'urdu': 'ur',
    'telugu': 'te',
    'indonesian': 'id',
}
