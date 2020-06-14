#!/bin/bash
if [ "$1" == "format" ]; then
	isort -rc ${@:2}
	yapf -ri --style .style.yapf ${@:2}
elif [ "$1" == "check" ]; then
	flake8 ${@:2}
	isort -rc --check-only --diff ${@:2}
	yapf -rd --style .style.yapf ${@:2}
	echo "check done!"
else
	echo "Unknown $1"
fi
