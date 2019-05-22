pwd

mkdir ../binary_segmentation
git clone .git ../binary_segmentation
cd  ../binary_segmentation
git checkout ReleaseDM1.0
cd ../maskrcnn


mkdir ../multiclass_segmentation
git clone .git ../multiclass_segmentation
cd  ../multiclass_segmentation
git checkout ReleaseDM1.1
cd ../maskrcnn


read -n1 -r -p "Press any key to continue..." key
