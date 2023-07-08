set image_name=terumo-service-multiple-models

REM Run Python files
call pipenv requirements > requirements.txt
call docker build -t %image_name% .
call docker tag %image_name% terumoapp/%image_name%:latest
call docker push terumoapp/%image_name%:latest



