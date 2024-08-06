#!/usr/bin/env python
"""
:Summary: Extracts the brain from clinical MRI and returns grey/white-matter mask.

:Description: This script performs bias field correction, brain extraction, intensity normalization, and generates masks for brain and intracranial volume.

:Requires: Python, NumPy, scipy, scikit-image, warnings, ANTs (Advanced Normalization Tools), Freesurfer

:TODO:

:AUTHOR: MDS
:ORGANIZATION: MGH/HMS
:CONTACT: mschirmer1@mgh.harvard.edu
:SINCE: 2024-03-20
:VERSION: 0.2
"""
#=============================================
# Metadata
#=============================================
__author__ = 'mds'
__contact__ = 'mschirmer1@mgh.harvard.edu'
__copyright__ = ''
__license__ = ''
__date__ = '2024-03'
__version__ = '0.2'

#=============================================
# Import statements
#=============================================
import sys
import os
from optparse import OptionParser
import re

import numpy as np
import scipy.ndimage as sn
import skimage.measure as skm
import sklearn.mixture
import warnings
import ants
import uuid
from subprocess import call

#import pdb

#=============================================
# Helper functions
#=============================================

def mean_shift_mode_finder(data, sigma=None, n_replicates=10, replication_method='percentiles', epsilon=None, max_iterations=1000, n_bins=None):
	"""
	Finds the mode of data using mean shift. Returns the best value and its score.

	e.g., (mean, score) = mean_shift_mode_finder(data.flatten())

	Inputs
	------
	data : one-dimensional ndarray
		Data to find the mode of.
	sigma : float, optional
		Kernel sigma (h) to be used; defaults to heuristic.
	n_replicates : int, optional
		How many times to run.
	replication_method : str, optional
		How to determine initialization for each replicate.
		'percentile' (uses n_replicate percentiles)
		'random' (uses n_replicate random valid values)
	epsilon : float, optional
		If the change in mode is less than this value, stop.
	max_iterations : int, optional
		Maximum number of iterations for each replicate.
	n_bins : int, optional
		How many bins to use for the data histogram.

	Adapted from 'meanShift.m' by Adrian Dalca and 'advanced_tools.py' by Ramesh Sridharan.
	"""

	if sigma is None:
		# Optimal bandwidth suggested by Bowman and Azzalini ('97) p31
		# adapted from ksr.m by Yi Cao
		sigma = np.median(np.abs(data-np.median(data))) / .6745 * (4./3./float(data.size))**0.2
	if epsilon is None:
		# heuristic
		epsilon = sigma / 100.
	if n_bins is None:
		n_bins = int(max(data.size / 10., 1))

	# Set up histogram
	dmin, dmax = data.min(), data.max()
	bins = np.linspace(dmin, dmax, n_bins)
	bin_size = (dmax - dmin) / (n_bins - 1.)
	(data_hist, _) = np.histogram(data, bins)
	bin_centers = bins[:-1] + .5 * bin_size

	# Set up replicates
	if replication_method == 'percentiles':
		if n_replicates > 1:
			percentiles = np.linspace(0, 100, n_replicates)
		else:
			percentiles = [50]

		inits = [np.percentile(data, p) for p in percentiles]

	elif replication_method == 'random':
		inits = np.random.uniform(data.min(), data.max(), n_replicates)

	scores = np.empty(n_replicates)
	means = np.empty(n_replicates)
	iter_counts = np.zeros(n_replicates) + np.inf
	# Core algorithm
	for i in range(n_replicates):
		mean = inits[i]
		change = np.inf
		for j in range(max_iterations):
			if change < epsilon:
				break
			weights = np.exp(-.5 * ((data - mean)/sigma) ** 2)
			assert weights.sum() != 0, "Weights sum to 0; increase sigma if appropriate (current val %f)" % sigma
			mean_old = mean
			mean = np.dot(weights, data) / float(weights.sum())
			change = np.abs(mean_old - mean)
			# print('%i, %f' %(j,change))

		if not j<(max_iterations-1):
			warnings.warn('Maximum number of iterations reached. %i' %max_iterations)
			print('Did not converge in replication %i/%i. Change: %f, Epsilon: %f, Iterations: %i' %(i+1, n_replicates, change, epsilon, max_iterations))

		means[i] = mean

		kernel = np.exp(-(bin_centers - mean)**2/(2*sigma**2))
		scores[i] = np.dot(kernel, data_hist)
		iter_counts[i] = j

	best = np.argmax(scores)
	n_good_replicates = np.sum(np.abs(means[best] - means) < sigma / 5.) - 1

	return (means[best], scores[best])

def get_biggest_connected_component(img):
	# Label connected components
	label_img = skm.label(img)

	# Find the biggest connected component (brain)
	volumes = [np.sum(label_img == label) for label in np.unique(label_img[label_img != 0])]
	brain_label = np.unique(label_img[label_img != 0])[np.argmax(volumes)]

	return label_img==brain_label

#=============================================
# Main method
#=============================================

def rescale(img, mask=None, new_intensity=0.75, mode=None):
	"""
	Normalizes image intensity and rescales it.

	Parameters
	----------
	img : ndarray
		Image data.
	mask : ndarray, optional
		Brain mask.
	new_intensity : float, optional
		Target intensity after normalization.
	mode : str, optional
		Normalization mode.

	Returns
	-------
	ndarray
		Rescaled image.
	float
		Normalization factor.
	"""

	# normalise to 0
	img = img.astype(np.float32)
	if mask is not None:
		assert mask.shape == img.shape, 'Mask is of different shape than image.'
	else:
		print('Estimating brain mask')
		prec=np.percentile(img, 5)
		mask = img>prec
	
	brain = np.multiply(img, mask)

	if mode=='percentile':
		# normalise 95%ile to 1000
		norm = np.mean(brain[brain > np.percentile(brain, 5)])
	else:
		# find mean shift mode
		(norm, score) = mean_shift_mode_finder(brain[brain>0.].flatten())

	# rescale intensity
	img = img * new_intensity/float(norm)

	return img, norm

def get_gmm_classes(img, mask, sequence='flair',wm_int=0.75):

	# blur the image a bit 
	img = sn.gaussian_filter(img, sigma=(0.5, 0.5, 0), order=0)

	# fit gaussian mixutre model with three components (in theory csf, wm, gm)
	gmm = sklearn.mixture.GaussianMixture(n_components=2).fit(img[mask>0].reshape(-1, 1))

	# get an estimate for the csf intensity
	# csf_int = np.median(img[img<0.375])
	csf_int = gmm.means_[np.argmax(np.abs(gmm.means_ - wm_int))].item()

	# get the scores of intensities between csf and wm
	xx = np.arange(np.min([csf_int, wm_int]),np.max([csf_int, wm_int]), 0.0001)
	zz = gmm.score_samples(xx.reshape(-1,1))

	# get minimum of the scores as threshold
	threshold = xx[np.argmin(zz)]

	# return updated mask
	if csf_int>wm_int:
		updated_mask = (img<threshold).astype(int)*mask	
	else:
		updated_mask = (img>threshold).astype(int)*mask

	return updated_mask

def main(argv):
	"""
	Main function to process MRI data.

	Parameters
	----------
	argv : argparse.Namespace
		Command-line arguments.
	"""

	infile = argv.i
	outfile = argv.o
	stats = []

	if argv.sub is None:
		argv.sub = infile
	stats.append(["ID","%s" %argv.sub])

	#############
	# check input and load data
	#############
	assert os.path.isfile(infile), "Input file %s not found." % infile
	mi=ants.image_read(infile)
	voxel_vol = np.prod(mi.spacing)
	if np.any(mi.numpy()<0):
		print("Some intensity values are negative. Please check input data (%s). Adjusting intensitise for now (+ min(intensities)). " %infile)
		mi = mi.new_image_like(mi.numpy() + np.min(mi.numpy()))


	#############
	# check output folder, create it if not, and define temporary file name
	#############
	outdir=os.path.dirname(outfile)
	if outdir=='':
		outdir=os.path.dirname(__file__)
	
	if not os.path.isdir(outdir):
		os.mkdir(outdir)

	#############
	# bias field correction
	#############	
	mi = ants.n4_bias_field_correction(mi)

	# get file name for bias corrected file
	if argv.bias is not None:
		biasfile=argv.bias
	else:
		biasfile=os.path.join(outdir, str(uuid.uuid4())+'.nii.gz')

	mi.to_filename(biasfile)
	#pdb.set_trace()

	#############
	# run synthstrip
	#############
	if argv.brain is not None:
		brainfile= argv.brain
	else:
		brainfile=os.path.join(outdir, str(uuid.uuid4())+'.nii.gz')

	if not os.path.isfile(brainfile):
		call(["mri_synthstrip", "-i", biasfile, "-m", brainfile, '-g',"-b", str(0)])
	mi_mask=ants.image_read(brainfile)
	stats.append(["Brain_volume", "%f" %(voxel_vol*np.sum(mi_mask.numpy()>0))])

	# had issues with q and s form differences that broke the following bias field correction
	ants.core.ants_image.copy_image_info(mi, mi_mask)

	# save brain mask if requested
	if argv.brain is not None:
		mi_mask.to_filename(argv.brain)

	#############
	# bias field correction after applying mask
	#############	
	mi = ants.n4_bias_field_correction(mi, mi_mask)

	# get file name for bias corrected file
	mi.to_filename(biasfile)

	#############
	# rescale image
	#############
	img, intnorm = rescale(mi.numpy(), mask=mi_mask.numpy())
	mi_norm = ants.image_clone(mi)
	mi_norm = mi_norm.new_image_like(img)
	stats.append(["NAWM_intensity", "%f" %(intnorm)])

	# save intensity normalized image if requested
	if argv.norm is not None:
		mi_norm.to_filename(argv.norm)

	#############
	# update gmwm mask
	#############
	# updated_mask = get_gmm_classes(mi_norm.numpy(), mi_mask.numpy())
	updated_mask = np.multiply(sn.gaussian_filter(mi_norm.numpy(), sigma=(0.75, 0.75, 0), order=0)>0.375, mi_mask.numpy())
	stats.append(["GMWM_volume", "%f" %(voxel_vol*np.sum(updated_mask))])

	#############
	# run synthseg to get ICV mask
	#############
	if argv.icv is not None:
		icvfile=argv.icv
		resampfile=os.path.join(outdir, str(uuid.uuid4())+'.nii.gz')
		if not os.path.isfile(argv.icv):
			call(["mri_synthseg", "--i", biasfile, "--parc","--robust","--threads",str(20), "--cpu", '--resample',resampfile,"--o", icvfile])

			# binarize segfile for icv mask
			segfile=ants.image_read(icvfile)
			segfile=ants.utils.threshold_image(segfile, 1e-15)
			# load resampled image and brain extract it for registration
			fi=ants.image_read(resampfile)
			fi = ants.utils.mask_image(fi, segfile)
			
			# mask patient image
			patient=ants.utils.mask_image(mi, mi_mask)
			
			# register upsampled image to patient space
			tx=ants.registration(fixed=patient, moving=fi, type_of_transformation='Affine')
			# get icv mask into patient space
			icv_mask = ants.apply_transforms(fixed=patient, moving=segfile, transformlist=tx['fwdtransforms'])
			
			# binarize mask
			icv_mask = ants.utils.threshold_image(icv_mask, 1e-15)

			# save icv mask
			icv_mask.to_filename(icvfile)
		else:
			icv_mask = ants.image_read(icvfile)
		
		stats.append(["ICV_volume", "%f" %(voxel_vol*np.sum(icv_mask.numpy()>0))])

		#############
		# update gmwm mask
		#############
		updated_mask = np.multiply(updated_mask, icv_mask.numpy())

	#############
	# save gmwm mask
	#############
	out = ants.image_clone(mi_mask)
	out = out.new_image_like(updated_mask)
	out.to_filename(outfile)

	#############
	# remove bias corrected file if not requested
	#############
	if argv.bias is None and os.path.isfile(biasfile):
		os.remove(biasfile)
	if argv.brain is None and os.path.isfile(brainfile):
		os.remove(brainfile)
	if argv.icv is not None and os.path.isfile(resampfile):
		os.remove(resampfile)

	#############
	# add info to stats file if requested
	#############
	if argv.stats is not None:
		import csv
		stats = np.array(stats).T.tolist()
		with open(argv.stats, 'w', newline='') as fid:
			writer = csv.writer(fid)
			writer.writerows(stats)

	return 0

if __name__ == "__main__":
	# catch input
	try:
		parser = OptionParser(description='Extraction of the brain from clincial MRI. Returns a grey/white-matter mask and optionally a brain and intracranial volume mask.', 
			epilog='Example: python ')
		parser.add_option('-i', dest='i', help='Input nifti image', metavar='FILE')
		parser.add_option('-o', dest='o', help='Output grey/white-matter mask as nifti', metavar='FILE')
		parser.add_option('--brain', dest='brain', help='Output brain mask', metavar='FILE', default=None)
		parser.add_option('--icv', dest='icv', help='Output icv mask', metavar='FILE', default=None)
		parser.add_option('--norm', dest='norm', help='Output intensity normalized brain image', metavar='FILE', default=None)
		parser.add_option('--bias', dest='bias', help='Output bias field corrected brain image', metavar='FILE', default=None)
		parser.add_option('--stats', dest='stats', help='Output stats file to save volumes and intensity normalization value', metavar='FILE', default=None)
		parser.add_option('--sub', dest='sub', help='Subject ID to be saved in stats file. If none is given, use input file', metavar='FILE', default=None)

		(options, args) = parser.parse_args()
	except:
		sys.exit()

main(options)
