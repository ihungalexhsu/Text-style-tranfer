python make_fastText_data.py ../../preprocess --dataset amazon
./fastText-0.2.0/fasttext supervised -input amazon/fastText.train -output amazon/model_amazon -lr 1.0 -epoch 25 -wordNgrams 2
./fastText-0.2.0/fasttext test amazon/model_amazon.bin amazon/fastText.dev
./fastText-0.2.0/fasttext test amazon/model_amazon.bin amazon/fastText.test
python make_fastText_data.py ../../preprocess --dataset yelp
./fastText-0.2.0/fasttext supervised -input yelp/fastText.train -output yelp/model_yelp -lr 0.9 -epoch 25 -wordNgrams 2
./fastText-0.2.0/fasttext test yelp/model_yelp.bin yelp/fastText.dev
./fastText-0.2.0/fasttext test yelp/model_yelp.bin yelp/fastText.test
python make_fastText_data.py ../../preprocess --dataset imagecaption
./fastText-0.2.0/fasttext supervised -input imagecaption/fastText.train -output imagecaption/model_imagecaption -lr 0.9 -epoch 20 -wordNgrams 2
./fastText-0.2.0/fasttext test imagecaption/model_imagecaption.bin imagecaption/fastText.dev
