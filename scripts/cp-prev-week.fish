#!/usr/local/bin/fish

set files .gitignore .vscode Readme.md data handler.py imagenet.py package-lock.json package.json requirements.txt scripts serverless.yml test utils.py

for file in $files
  cp -R $argv[1]/$file $argv[2]
end

cd $argv[2]

# install all necessary modules
npm install

serverless
