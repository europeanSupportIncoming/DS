#!/usr/bin/env bash

name="$1"

if [ -z "$name" ]; then
	read -p "your name is? " name
fi

echo "Hello, $name!"
