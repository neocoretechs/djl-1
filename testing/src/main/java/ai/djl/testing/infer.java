package ai.djl.testing;
import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.Artifact;
import ai.djl.repository.Artifact.Item;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
public class infer {
	private infer() {}
	   //public static void main(String[] args) throws IOException, ModelException, TranslateException {
	     //   DetectedObjects detection = infer.predict();
	     //   System.out.println(detection);
	   // }

		public static void showEnv() throws ModelNotFoundException, IOException {
	        Engine.getAllEngines().forEach(e->{
	        	System.out.println("Engine:"+e);
	        });
	        System.out.println("Default engine name:"+Engine.getDefaultEngineName());
	        Map<Application,List<Artifact>> m = ModelZoo.listModels();
	        m.forEach((k, v) -> {
	        	System.out.println(" App:"+k);
	        	v.forEach((la)->{
	        	System.out.println("Name:"+((Artifact)la).getName());
	        	Map<String,Object> args = ((Artifact)la).getArguments();
	        	args.forEach((ka,va)->System.out.println("arg:"+ka+","+va));
	        	Map<String,Item> fils = ((Artifact)la).getFiles();
	        	fils.forEach((kf,vf)->System.out.println("Files:"+kf+": "+vf.getName()+" type:"+vf.getType()+" size:"+vf.getSize()+" ext:"+vf.getExtension()+" uri:"+vf.getUri()));
	        	Map<String,String> props = ((Artifact)la).getProperties();
	        	props.forEach((kp,vp)->System.out.println("Prop:"+kp+", "+vp));
	        	});
	        });
		}
		

	    
	    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
	            throws IOException {
	            Path outputDir = Paths.get("build/output");
	            Files.createDirectories(outputDir);

	            img.drawBoundingBoxes(detection);

	            Path imagePath = outputDir.resolve("detected.png");
	            // OpenJDK can't save jpg with alpha channel
	            img.save(Files.newOutputStream(imagePath), "png");
	        }

	        public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
	        	showEnv();
	            int imageSize = 640;
	            Pipeline pipeline = new Pipeline();
	            pipeline.add(new Resize(imageSize));
	            pipeline.add(new ToTensor());

	           // List<String> synset = new ArrayList<>(80);
	            Path myPath = Paths.get("yolo/coco_80_labels_list.txt");
	            List<String> synset = Files.readAllLines(myPath, StandardCharsets.UTF_8);
	            synset.forEach(line -> System.out.println(line));
	            //for (int i = 0; i < 80; i++) {
	                //synset.add("Person");
	            //}

	            Translator<Image, DetectedObjects> translator =  YoloV5Translator
	                .builder()
	                .setPipeline(pipeline)
	                .optSynset(synset)
	                .optThreshold(0.2f)
	                .build();

	            Criteria<Image, DetectedObjects> criteria = Criteria.builder()
	                .setTypes(Image.class, DetectedObjects.class)
	                .optModelUrls("yolo/")
	                .optModelName("yolov5s.1.torchscript")
	                .optTranslator(translator)
	                .optProgress(new ProgressBar())
	                .optEngine("PyTorch")
	                .build();

		        //Path imageFile = Paths.get("C:/Users/groff/Downloads/djl-0.20.0/djl-0.20.0/examples/src/test/resources/dog_bike_car.jpg");
		        //Image input = ImageFactory.getInstance().fromFile(imageFile);
		      //Image input = ImageFactory.getInstance().fromUrl("https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg");
		        Path imageFile = Paths.get("C:/etc/images/pexels-photo-1029597.jpeg");
		        Image input = ImageFactory.getInstance().fromFile(imageFile);
	            try(ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
	                try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
	                    DetectedObjects objects = predictor.predict(input);
	                    List<BoundingBox> boxes = new ArrayList<>();
	                    List<String> names = new ArrayList<>();
	                    List<Double> prob = new ArrayList<>();
	                    System.out.println("Number of objects:"+objects.getNumberOfObjects());
	                    for (Classifications.Classification obj : objects.items()) {
	                        DetectedObjects.DetectedObject objConvered = (DetectedObjects.DetectedObject) obj;
	                        BoundingBox box = objConvered.getBoundingBox();
	                        Rectangle rec = box.getBounds();
	                        Rectangle rec2 = new Rectangle(
	                            rec.getX() / 640,
	                            rec.getY() / 640,
	                            rec.getWidth() / 640,
	                            rec.getHeight() / 640
	                            );
	                        boxes.add(rec2);
	                        names.add(obj.getClassName()+" "+((int)(obj.getProbability()*100.0f))+"%");
	                        prob.add(obj.getProbability());
	                        System.out.println("added "+obj.getClassName()+" "+obj.getProbability());
	                    }
	                    DetectedObjects converted = new DetectedObjects(names, prob, boxes);
	                    saveBoundingBoxImage(input, converted);
	                }
	            }
	            
	        }
}
