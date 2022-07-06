git clone https://github.com/openai/baselines.git
cd baselines
git fetch origin pull/620/head:chunk
git checkout chunk
pip install -e .
cd ..
git clone https://github.com/eleramp/pybullet-object-models.git
pip3 install -e pybullet-object-models/
pip3 install -e shadowhand_gym/