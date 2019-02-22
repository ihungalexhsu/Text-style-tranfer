python preprocessing.py --dataset yelp --min_word_freq 5 --max_vocab_number 10000 --max_len 15 ../raw_data
python preprocessing.py --dataset amazon --min_word_freq 3 --max_vocab_number 60000 --max_len 20 ../raw_data
python preprocessing.py --dataset imagecaption --min_word_freq 0 --max_vocab_number 60000 --max_len 20 ../raw_data

