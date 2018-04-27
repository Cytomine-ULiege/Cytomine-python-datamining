import tempfile
from argparse import ArgumentParser

import os
import cv2
import numpy as np
from cytomine import Cytomine, CytomineJob
from cytomine.models import AlgoAnnotationTerm
from cytomine_sldc import CytomineSlide, CytomineTileBuilder
from shapely.affinity import affine_transform, translate
from sklearn.utils import check_random_state
from sldc import DispatchingRule, ImageWindow, Loggable, Logger, Segmenter, StandardOutputLogger, SLDCWorkflowBuilder

from pyxit_classifier import PyxitClassifierAdapter


def _upload_annotation(cytomine, img_inst, polygon, label=None, proba=1.0):
    """Upload an annotation and its term (if provided)"""
    image_id = img_inst.id

    # Transform polygon to match cytomine (bottom-left) origin point
    polygon = affine_transform(polygon, [1, 0, 0, -1, 0, img_inst.height])

    annotation = cytomine.add_annotation(polygon.wkt, image_id)
    if label is not None and annotation is not None:
        cytomine.add_annotation_term(annotation.id, label, label, proba, annotation_term_model=AlgoAnnotationTerm)


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


def main(argv):
    with CytomineJob.from_cli(argv) as job:
        if not os.path.exists(job.parameters.working_path):
            os.makedirs(job.parameters.working_path)

        # create workflow component
        logger = StandardOutputLogger(Logger.INFO)
        random_state = check_random_state(job.parameters.rseed)
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

        slide = CytomineSlide(job.parameters.cytomine_id_image)
        results = workflow.process(slide)

        # Upload results
        for polygon, dispatch, cls, proba in results:
            if cls is not None:
                # if image is a window, the polygon must be translated
                if isinstance(slide, ImageWindow):
                    polygon = translate(polygon, slide.abs_offset_x, slide.abs_offset_y)
                # actually upload the annotation
                _upload_annotation(
                    self._cytomine,
                    slide.image_instance,
                    polygon,
                    label=cls,
                    proba=proba
                )


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
