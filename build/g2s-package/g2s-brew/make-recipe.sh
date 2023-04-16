version_number=$(cat ../../../version)

COMMIT_HASH256=$(curl  -sL $( echo "https://github.com/GAIA-UNIL/G2S/archive/refs/tags/vx.y.z.tar.gz" | sed  -e "s/x.y.z/${version_number}/g")  |  shasum -a 256 | cut -d " " -f 1 )
COMMIT_HASH=$(curl -s https://api.github.com/repos/GAIA-UNIL/g2s/commits/moveComputationFileTo-tmp | grep sha | head -n 1 | awk '{print $2}' | sed 's/\"//g' | sed 's/,//g')
COMMIT_HASH256=$(curl  -sL https://github.com/GAIA-UNIL/g2s/archive/${COMMIT_HASH}.tar.gz  |  shasum -a 256 | cut -d " " -f 1 )

sed -i -e "s/COMMIT_HASH256/${COMMIT_HASH256}/g" ./g2s.rb
sed -i -e "s/COMMIT_HASH/${COMMIT_HASH}/g" ./g2s.rb
sed -i -e "s/x.y.z/${version_number}/g" ./g2s.rb