
docker run --rm -d -t --name=test -p 8888:8888 --mount src="$(pwd)",target=/myapp,type=bind test_python 
docker exec -ti test bash

jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser --allow-root


