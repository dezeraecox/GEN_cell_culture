//This macro batch processes all the files in a folder and any
// subfolders in that folder. For only red or green images, comment
//out the appropriate sections in the last function

// initialises all functions (count files, process files)
dir = getDirectory("Choose a Directory ");
//setBatchMode(true);
count = 0;
countFiles(dir);
n = 0;
processFiles(dir);
saveAs("Results", dir+"_Measurements.csv");
print(count+" files processed");
print("Done");

//produces a list of all files in the directory
function countFiles(dir) {
   list = getFileList(dir);
   for (i=0; i<list.length; i++) {
       if (endsWith(list[i], "/"))
           countFiles(""+dir+list[i]);
       else
           count++;
   }
}


//processes all files in both current directory and subfolders
function processFiles(dir) {
   list = getFileList(dir);
   for (i=0; i<list.length; i++) {
       showProgress(n++, count);
       path = dir+list[i];
       processFile(path);
       }
   }


//applies the relevant actions to each file
function processFile(path) {
    if (endsWith(path, ".tif")) {
        open(path);
        imageTitle=getTitle();//returns a string with the image title
        imageSorter(imageTitle);


   }

function imageSorter(imageTitle){
		image_name = split(imageTitle,"-");
		outputDir = getDirectory("image") + "/Results/";

		selectWindow(imageTitle);
		//setMinAndMax(1500, 2800);
		//run("Apply LUT");
		setAutoThreshold("Default"); //or "Moments"
		setOption("BlackBackground", true);
		run("Convert to Mask");
		run("Gaussian Blur...", "sigma=6");
		run("Set Measurements...", "area mean standard min area_fraction limit display add nan");
		run("Measure");
		saveAs("Tiff", outputDir+image_name[0]+"_Mask.csv");


     	while (nImages>0) {
          selectImage(nImages);
          close();

}
}
}
