import React, { useEffect } from 'react';
import { Grid } from '@fluentui/react-northstar';
import { useDispatch, useSelector } from 'react-redux';

import { State } from '../../store/State';
import { Part } from '../../store/part/partTypes';
import { thunkGetCapturedImages, addCapturedImages } from '../../store/part/partActions';
import LabelingPageDialog from '../LabelingPageDialog';
import LabelDisplayImage from '../LabelDisplayImage';

export const UploadPhotos = ({ partId }): JSX.Element => {
  const dispatch = useDispatch();
  const { capturedImages } = useSelector<State, Part>((state) => state.part);

  useEffect(() => {
    dispatch(thunkGetCapturedImages());
  }, [dispatch]);

  function handleUpload(e: React.ChangeEvent<HTMLInputElement>): void {
    for (let i = 0; i < e.target.files.length; i++) {
      const formData = new FormData();
      formData.append('image', e.target.files[i]);
      formData.append('part', `http://localhost:8000/api/parts/${partId}/`);
      fetch(`/api/images/`, {
        method: 'POST',
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => dispatch(addCapturedImages(data)))
        .catch((err) => console.error(err));
    }
  }

  return (
    <>
      <input type="file" onChange={handleUpload} accept="image/*" multiple />
      <CapturedImagesContainer capturedImages={capturedImages} />
    </>
  );
};

const CapturedImagesContainer = ({ capturedImages }): JSX.Element => {
  return (
    <Grid columns="2" styles={{ border: '1px solid grey', height: '100%', gridGap: '10px', padding: '10px' }}>
      {capturedImages.map((image, i) => (
        <LabelingPageDialog
          key={i}
          imageIndex={i}
          trigger={
            <LabelDisplayImage imgSrc={image.image} pointerCursor width={300} height={150} imgPadding="0" />
          }
        />
      ))}
    </Grid>
  );
};
