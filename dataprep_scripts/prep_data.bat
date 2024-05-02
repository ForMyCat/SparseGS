:: Create a folder named image_p and put your selected images in it
:: Enter the sparse folder and run the following sequentially
mkdir sparse\1
cmd /c colmap model_converter --input_path sparse\0 --output_path sparse\1 --output_type TXT
cmd /c python C:\Users\27270\docker_env\3dgs_scripts\scripts\subselect_imgs.py sparse\1\images.txt images_p
cmd /c colmap feature_extractor --ImageReader.single_camera 1  --ImageReader.camera_model OPENCV --database_path db_p.db --image_path images_p > extractor_output.txt
cmd /c python C:\Users\27270\docker_env\3dgs_scripts\scripts\reindex_id.py sparse\1\images.txt
cmd /c colmap exhaustive_matcher --database_path db_p.db
mkdir sparse\2
cmd /c colmap point_triangulator --database_path db_p.db --image_path images_p --input_path sparse\1 --output_path sparse\2
cmd /c colmap image_undistorter --image_path images_p --input_path sparse\2 --output_path images_und
mkdir images_und\sparse\0
cmd /c colmap model_converter --input_path .\images_und\sparse --output_path .\images_und\sparse\0 --output_type TXT
:: Now images_und is your entire prepared folder, rename it to your experiment name and you are ready to go