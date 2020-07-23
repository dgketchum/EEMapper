
import datetime
import logging

import ee
from dateutil.relativedelta import *

from map.openet import utils


def custom(target_coll, source_coll, interp_days=32,
           use_joins=False, compute_product=False):
    """Interpolate non-daily source images to a daily target image collection
    Parameters
    ----------
    target_coll : ee.ImageCollection
        Source images will be interpolated to each target image time_start.
        Target images should have a daily time step.  This will typically be
        the reference ET (ETr) collection.
    source_coll : ee.ImageCollection
        Images that will be interpolated to the target image collection.
        This will typically be the fraction of reference ET (ETrF) collection.
    interp_days : int, optional
        Number of days before and after each image date to include in the
        interpolation (the default is 32).
    use_joins : bool, optional
        If True, the source collection will be joined to the target collection
        before mapping/interpolation and the source images will be extracted
        from the join properties ('prev' and 'next').
        Setting use_joins=True should be more memory efficient.
        If False, the source images will be built by filtering the source
        collection separately for each image in the target collection
        (inside the mapped function).
    compute_product : bool, optional
        If True, compute the product of the target and all source image bands.
        The default is False.
    Returns
    -------
    ee.ImageCollection() of daily interpolated images
    Raises
    ------
    ValueError
        If `interp_method` is not a supported method.
    """

    prev_filter = ee.Filter.And(
        ee.Filter.maxDifference(
            difference=(interp_days + 1) * 24 * 60 * 60 * 1000,
            leftField='system:time_start',
            rightField='system:time_start',
        ),
        ee.Filter.greaterThan(
            leftField='system:time_start',
            rightField='system:time_start',
        )
    )

    next_filter = ee.Filter.And(
        ee.Filter.maxDifference(
            difference=(interp_days + 1) * 24 * 60 * 60 * 1000,
            leftField='system:time_start',
            rightField='system:time_start',
        ),
        ee.Filter.lessThanOrEquals(
            leftField='system:time_start',
            rightField='system:time_start',
        )
    )

    if use_joins:
        # Join the neighboring Landsat images in time
        target_coll = ee.ImageCollection(
            ee.Join.saveAll(
                matchesKey='prev',
                ordering='system:time_start',
                ascending=True,
                outer=True,
            ).apply(
                primary=target_coll,
                secondary=source_coll,
                condition=prev_filter,
            )
        )

        target_coll = ee.ImageCollection(
            ee.Join.saveAll(
                matchesKey='next',
                ordering='system:time_start',
                ascending=False,
                outer=True,
            ).apply(
                primary=target_coll,
                secondary=source_coll,
                condition=next_filter,
            )
        )

    def _linear(image):
        """Linearly interpolate source images to target image time_start(s)
        Parameters
        ----------
        image : ee.Image.
            The first band in the image will be used as the "target" image
            and will be returned with the output image.
        Returns
        -------
        ee.Image of interpolated values with band name 'src'
        Notes
        -----
        The source collection images must have a time band.
        This function is intended to be mapped over an image collection and
            can only take one input parameter.
        """
        # target_img = ee.Image(image).select(0).double()
        target_date = ee.Date(image.get('system:time_start'))

        # All filtering will be done based on 0 UTC dates
        utc0_date = utils.date_0utc(target_date)
        # utc0_time = target_date.update(hour=0, minute=0, second=0)\
        #     .millis().divide(1000).floor().multiply(1000)
        time_img = ee.Image.constant(utc0_date.millis()).double()

        # Build nodata images/masks that can be placed at the front/back of
        #   of the qm image collections in case the collections are empty.
        bands = source_coll.first().bandNames()
        prev_qm_mask = ee.Image.constant(ee.List.repeat(1, bands.length())) \
            .double().rename(bands).updateMask(0) \
            .set({
            'system:time_start': utc0_date.advance(
                -interp_days - 1, 'day').millis()})
        next_qm_mask = ee.Image.constant(ee.List.repeat(1, bands.length())) \
            .double().rename(bands).updateMask(0) \
            .set({
            'system:time_start': utc0_date.advance(
                interp_days + 2, 'day').millis()})

        if use_joins:
            # Build separate mosaics for before and after the target date
            prev_qm_img = ee.ImageCollection \
                .fromImages(ee.List(ee.Image(image).get('prev'))) \
                .merge(ee.ImageCollection(prev_qm_mask)) \
                .sort('system:time_start', True) \
                .mosaic()
            next_qm_img = ee.ImageCollection \
                .fromImages(ee.List(ee.Image(image).get('next'))) \
                .merge(ee.ImageCollection(next_qm_mask)) \
                .sort('system:time_start', False) \
                .mosaic()
        else:
            # Build separate collections for before and after the target date
            prev_qm_coll = source_coll \
                .filterDate(utc0_date.advance(-interp_days, 'day'), utc0_date) \
                .merge(ee.ImageCollection(prev_qm_mask))
            next_qm_coll = source_coll \
                .filterDate(utc0_date, utc0_date.advance(interp_days + 1, 'day')) \
                .merge(ee.ImageCollection(next_qm_mask))

            # Flatten the previous/next collections to single images
            # The closest image in time should be on "top"
            # CGM - Is the previous collection already sorted?
            # prev_qm_img = prev_qm_coll.mosaic()
            prev_qm_img = prev_qm_coll.sort('system:time_start', True) \
                .mosaic()
            next_qm_img = next_qm_coll.sort('system:time_start', False) \
                .mosaic()

        # DEADBEEF - It might be easier to interpolate all bands instead of
        #   separating the value and time bands
        # prev_value_img = ee.Image(prev_qm_img).double()
        # next_value_img = ee.Image(next_qm_img).double()

        # Interpolate all bands except the "time" band
        prev_bands = prev_qm_img.bandNames() \
            .filter(ee.Filter.notEquals('item', 'time'))
        next_bands = next_qm_img.bandNames() \
            .filter(ee.Filter.notEquals('item', 'time'))
        prev_value_img = ee.Image(prev_qm_img.select(prev_bands)).double()
        next_value_img = ee.Image(next_qm_img.select(next_bands)).double()
        prev_time_img = ee.Image(prev_qm_img.select('time')).double()
        next_time_img = ee.Image(next_qm_img.select('time')).double()

        # Fill masked values with values from the opposite image
        # Something like this is needed to ensure there are always two
        #   values to interpolate between
        # For data gaps, this will cause a flat line instead of a ramp
        prev_time_mosaic = ee.Image(ee.ImageCollection.fromImages([
            next_time_img, prev_time_img]).mosaic())
        next_time_mosaic = ee.Image(ee.ImageCollection.fromImages([
            prev_time_img, next_time_img]).mosaic())
        prev_value_mosaic = ee.Image(ee.ImageCollection.fromImages([
            next_value_img, prev_value_img]).mosaic())
        next_value_mosaic = ee.Image(ee.ImageCollection.fromImages([
            prev_value_img, next_value_img]).mosaic())

        # Calculate time ratio of the current image between other cloud free images
        time_ratio_img = time_img.subtract(prev_time_mosaic) \
            .divide(next_time_mosaic.subtract(prev_time_mosaic))

        # Interpolate values to the current image time
        interp_img = next_value_mosaic.subtract(prev_value_mosaic) \
            .multiply(time_ratio_img).add(prev_value_mosaic)

        # Pass the target image back out as a new band
        target_img = image.select([0]).double()
        output_img = interp_img.addBands([target_img]) \
\
        # TODO: Come up with a dynamic way to name the "product" bands
        # The product bands will have a "_1" appended to the name
        # i.e. "et_fraction" -> "et_fraction_1"
        if compute_product:
            output_img = output_img \
                .addBands([interp_img.multiply(target_img)])

        return output_img.set({
            'system:index': image.get('system:index'),
            'system:time_start': image.get('system:time_start'),
            # 'system:time_start': utc0_time,
        })

    interp_coll = ee.ImageCollection(target_coll.map(_linear))

    return interp_coll


def aggregate_to_daily(image_coll, start_date=None, end_date=None,
                       agg_type='mean'):
    """Aggregate images by day without using joins
    The primary purpose of this function is to join separate Landsat images
    from the same path into a single daily image.
    Parameters
    ----------
    image_coll : ee.ImageCollection
        Input image collection.
    start_date :  date, number, string, optional
        Start date.
        Needs to be an EE readable date (i.e. ISO Date string or milliseconds).
    end_date :  date, number, string, optional
        Exclusive end date.
        Needs to be an EE readable date (i.e. ISO Date string or milliseconds).
    agg_type : {'mean'}, optional
        Aggregation type (the default is 'mean').
        Currently only a 'mean' aggregation type is supported.
    Returns
    -------
    ee.ImageCollection()
    Notes
    -----
    This function should be used to mosaic Landsat images from same path
        but different rows.
    system:time_start of returned images will be 0 UTC (not the image time).
    """
    if start_date and end_date:
        test_coll = image_coll.filterDate(ee.Date(start_date), ee.Date(end_date))
    elif start_date:
        test_coll = image_coll.filter(ee.Filter.greaterThanOrEquals(
            'system:time_start', ee.Date(start_date).millis()))
    elif end_date:
        test_coll = image_coll.filter(ee.Filter.lessThan(
            'system:time_start', ee.Date(end_date).millis()))
    else:
        test_coll = image_coll

    # Build a sorted list of the unique "dates" in the image_coll
    date_list = ee.List(test_coll.aggregate_array('system:time_start')) \
        .map(lambda time: ee.Date(ee.Number(time)).format('yyyy-MM-dd')) \
        .distinct().sort()

    def aggregate_func(date_str):
        start_date = ee.Date(ee.String(date_str))
        end_date = start_date.advance(1, 'day')
        agg_coll = image_coll.filterDate(start_date, end_date)

        if agg_type.lower() == 'mean':
            agg_img = agg_coll.mean()
        # elif agg_type.lower() == 'median':
        #     agg_img = agg_coll.median()
        else:
            raise ValueError(f'unsupported agg_type "{agg_type}"')

        return agg_img.set({
            'system:index': start_date.format('yyyyMMdd'),
            'system:time_start': start_date.millis(),
            'date': start_date.format('yyyy-MM-dd'),
        })

    return ee.ImageCollection(date_list.map(aggregate_func))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
