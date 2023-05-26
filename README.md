## Getting started

# Installation

For a self-contained installation, follow the following instructions.

Tested on python3.8+.
```
python3 -m venv ltl-env
source ltl-env/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## LTL to Buchi Requirement 1:
Download SPOT and unzip to the current directory:
https://spot.lrde.epita.fr/install.html

Install via:
```
dir=$PWD
mkdir $dir/ltl-env/spot
cd spot-2.10.4
./configure --prefix $dir
make -j8
make install -j8
cp -r $dir/lib/python3.10/site-packages/* $dir/ltl-env/lib/python3.10/site-packages/
rm spot-2.10.4.tar.gz 
```

## LTL to Buchi Requirement 2:
Download and unzip Rabinizer to current directory:
https://www7.in.tum.de/~kretinsk/rabinizer4.html

Note: Must have java >8 installed as well to run Rabinizer.

# Run:

```
python3 run.py <YOUR CONFIGURATION HERE, Stored in ./cfgs/>.yaml --restart
```
For example:
```
python3 run.py flatworld_continuous.yaml --restart
```
