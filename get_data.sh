mkdir -p data
cd data
wget -nc 'http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip'
unzip -o 'SciTailV1.1.zip'
wget -nc 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
unzip -o 'snli_1.0.zip'
wget -nc 'https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'
unzip -o 'multinli_1.0.zip'
