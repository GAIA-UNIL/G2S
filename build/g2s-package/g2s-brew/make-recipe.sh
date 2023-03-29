version_number="0.95"  #$(cat ../../../version)

COMMIT_HASH256=$(curl  -sL $( echo "https://github.com/GAIA-UNIL/G2S/archive/refs/tags/x.y.z.tar.gz" | sed  -e "s/x.y.z/${version_number}/g")  |  shasum -a 256 | cut -d " " -f 1 )

sed -i -e "s/COMMIT_HASH256/${COMMIT_HASH256}/g" ./g2s.rb
sed -i -e "s/x.y.z/${version_number}/g" ./g2s.rb