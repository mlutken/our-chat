English Dictionaries
====================
 - **Dictionary JSON, words with meanings:** https://github.com/nightblade9/simple-english-dictionary/tree/main
 - **209.785 words list**: https://github.com/SteveKein/English-Words-Dictionary#
 - **A list of 40,940 nouns** https://gist.github.com/trag1c/f74b2ab3589bc4ce5706f934616f6195


Compact json to one record per line
===================================
 - jq -c '.[]' _myfile.json  > myfile.json
 - jq '.[]' _myfile.json | sed '$!s/$/,/' | sed '1s/^/[/' | sed '$s/$/]/' > myfile.json

