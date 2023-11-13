package name.mjw.djl;

import java.io.IOException;
import java.nio.file.Paths;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

public class ResNetExample {

	public static void main(String[] args)
			throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {

		DownloadUtils.download(
				"https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/resnet/0.0.1/traced_resnet18.pt.gz",
				"build/pytorch_models/resnet18/resnet18.pt", new ProgressBar());

		DownloadUtils.download(
				"https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/synset.txt",
				"build/pytorch_models/resnet18/synset.txt", new ProgressBar());

		Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
				.addTransform(new Resize(256))
				.addTransform(new CenterCrop(224, 224))
				.addTransform(new ToTensor())
				.addTransform(new Normalize(new float[] { 0.485f, 0.456f, 0.406f }, new float[] { 0.229f, 0.224f, 0.225f }))
				.optApplySoftmax(true).build();

		// this model requires mapLocation for GPU
		Criteria<Image, Classifications> criteria = Criteria.builder()
				.setTypes(Image.class, Classifications.class)
				.optModelPath(Paths.get("build/pytorch_models/resnet18"))
				.optOption("mapLocation", "true")
				.optTranslator(translator)
				.optProgress(new ProgressBar())
				.build();

		ZooModel<Image, Classifications> model = criteria.loadModel();

		Image img = ImageFactory.getInstance()
				.fromUrl("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg");
		img.getWrappedImage();

		Predictor<Image, Classifications> predictor = model.newPredictor();
		Classifications classifications = predictor.predict(img);

		System.out.println(classifications);
	}
}
