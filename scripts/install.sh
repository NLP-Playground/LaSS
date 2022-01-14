pip3 install numpy

pip3 install -U portalocker

git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 6f847c8654d56b4d1b1fbacec027f47419426ddb
pip3 install -e .
cd ..
