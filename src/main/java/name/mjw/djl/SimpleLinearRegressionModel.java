package name.mjw.djl;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class SimpleLinearRegressionModel {

	public static void main(String[] args)
			throws MalformedModelException, IOException, NumberFormatException, TranslateException {
		Path modelDir = Paths.get("./src/main/resources/name/mjw/djl");
		Model model = Model.newInstance("model1.zip");
		model.load(modelDir);

		System.out.println(model);

		/*
		 * See https://towardsdatascience.com/pytorch-model-in-deep-java-library-
		 * a9ca18d8ce51
		 */
		Translator<Float, Float> translator = new Translator<Float, Float>() {
			@Override
			public NDList processInput(TranslatorContext ctx, Float input) {
				NDManager manager = ctx.getNDManager();
				NDArray array = manager.create(new float[] { input });
				return new NDList(array);
			}

			@Override
			public Float processOutput(TranslatorContext ctx, NDList list) {
				NDArray tempArr = list.get(0);
				return tempArr.getFloat();
			}

			@Override
			public Batchifier getBatchifier() {
				// The Batchifier describes how to combine a batch together
				// Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a
				// single [N, X1, X2, ...] array
				return Batchifier.STACK;
			}

		};

		Predictor<Float, Float> predictor = model.newPredictor(translator);
		System.out.println(predictor.predict(Float.valueOf("0.2")));

	}
}
