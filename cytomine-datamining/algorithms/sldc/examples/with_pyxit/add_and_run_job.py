import tempfile
from argparse import ArgumentParser

import os
import cv2
import numpy as np
from cytomine import Cytomine, CytomineJob
from cytomine.models import AlgoAnnotationTerm, Annotation
from cytomine_sldc import CytomineSlide, CytomineTileBuilder
from shapely.affinity import affine_transform, translate
from sklearn.utils import check_random_state
from sldc import DispatchingRule, ImageWindow, Loggable, Logger, Segmenter, StandardOutputLogger, SLDCWorkflowBuilder, \
    PolygonClassifier

from pyxit_classifier import PyxitClassifierAdapter


class DemoSegmenter(Segmenter):
    def __init__(self, threshold):
        """A simple segmenter that performs a simple thresholding on the Green channel of the image"""
        super(DemoSegmenter, self).__init__()
        self._threshold = threshold

    def segment(self, image):
        mask = np.array(image[:, :, 1] < self._threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask * 255


class ValidAreaRule(DispatchingRule):
    def __init__(self, min_area):
        """A rule which matches polygons of which the area is greater than min_area"""
        super(ValidAreaRule, self).__init__()
        self._min_area = min_area

    def evaluate(self, image, polygon):
        return self._min_area < polygon.area


# TODO tmp
class ConstantClassifier(PolygonClassifier):
    def predict(self, image, polygon):
        return self._label

    def __init__(self, label=526928):
        self._label = label


def main(argv):
    with CytomineJob.from_cli(argv) as job:
        if not os.path.exists(job.parameters.working_path):
            os.makedirs(job.parameters.working_path)

        # create workflow component
        logger = StandardOutputLogger(Logger.INFO)
        random_state = check_random_state(int(job.parameters.rseed))
        tile_builder = CytomineTileBuilder(working_path=job.parameters.working_path)
        segmenter = DemoSegmenter(job.parameters.threshold)
        area_rule = ValidAreaRule(job.parameters.min_area)
        classifier = PyxitClassifierAdapter.build_from_pickle(
            job.parameters.pyxit_model_path, tile_builder, logger,
            random_state=random_state,
            n_jobs=job.parameters.n_jobs,
            working_path=job.parameters.working_path
        )

        builder = SLDCWorkflowBuilder()
        builder.set_n_jobs(job.parameters.n_jobs)
        builder.set_logger(logger)
        builder.set_overlap(job.parameters.sldc_tile_overlap)
        builder.set_tile_size(job.parameters.sldc_tile_width, job.parameters.sldc_tile_height)
        builder.set_tile_builder(tile_builder)
        builder.set_segmenter(segmenter)
        builder.add_classifier(area_rule, classifier, dispatching_label="valid")
        workflow = builder.get()

        slide = CytomineSlide(job.parameters.cytomine_image_id)
        results = workflow.process(slide)

        # Upload results
        for polygon, label, proba, dispatch in results:
            if label is not None:
                # if image is a window, the polygon must be translated
                if isinstance(slide, ImageWindow):
                    polygon = translate(polygon, slide.abs_offset_x, slide.abs_offset_y)
                # upload the annotation
                polygon = affine_transform(polygon, [1, 0, 0, -1, 0, slide.image_instance.height])
                annotation = Annotation(location=polygon.wkt, id_image=slide.image_instance.id).save()
                AlgoAnnotationTerm(id_annotation=annotation.id, id_term=label, rate=float(proba)).save()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
