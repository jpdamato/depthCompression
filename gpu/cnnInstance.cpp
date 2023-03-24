
#include "cnnInstance.h"
#include "../include/darknet.h"
#include "yoloopenCL.h"
#include "cl_utils.h"
#include "cl_MemHandler.h"

#include "../u_ProcessTime.h"



#ifdef TRACKING_LIB
using namespace trackingLib;
#endif

char **_names;
image **_alphabet;


image make_empty_imageX(int w, int h, int c)
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}


image make_imageX(int w, int h, int c)
{
	image out = make_empty_imageX(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	return out;
}


cv::Mat GetSquareImage(const cv::Mat& img, int target_width)
{
	int width = img.cols,
		height = img.rows;

	cv::Mat square = cv::Mat::zeros(target_width, target_width, img.type());

	int max_dim = (width >= height) ? width : height;
	float scale = ((float)target_width) / max_dim;
	cv::Rect roi;
	if (width >= height)
	{
		roi.width = target_width;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = (target_width - roi.height) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = target_width;
		roi.width = width * scale;
		roi.x = (target_width - roi.width) / 2;
	}

	cv::resize(img, square(roi), roi.size());

	return square;
}

image mat_to_imageX(cv::Mat& m, image& im)
{


	IplImage ipl = m;
	IplImage* src = &ipl;
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	unsigned char *data = (unsigned char *)src->imageData;
	int step = src->widthStep;
	int i, j, k;
//first clean
	for (i = 0; i < h*c*w; i++) 
		im.data[i] = 0.0f;
// now update
	for (i = 0; i < h; i++) {
		for (k = 0; k < c; k++) {
			for (j = 0; j < w; j++) {
				im.data[k*w*h + i * w + j] = data[i*step + j * c + k] / 255.;
			}
		}
	}
	return im;
}

std::string ExePath() {
/*	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos);
	*/

	return "D:\\Sdks\\Yolo\\darknet\\x64\\Release\\";
}

std::string randomString() {

	int len = 12;
	char s[13];
	static const char alphanum[] =
		"0123456789"
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz";

	for (int i = 0; i < len; ++i) {
		s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
	}

	s[len] = 0;
	return std::string(s);
}



#ifdef TRACKING_LIB

trackingLib::Blob* insideAnyBlob(trackingLib::BlobsByFrame* bb0, cv::Point2f pt, float offset)
{

	for (auto b : bb0->trackBlobs)
	{
		Rect r = b->getRect();

		if ((pt.x >= r.x - offset) && (pt.x <= r.x + r.width + offset) &&
			(pt.y >= r.y - offset) && (pt.y <= r.y + r.height + offset))
		{
			return b;
		}

	}

	return NULL;
}

BlobsByFrame* _converToBlobs(cv::Mat& input, detection* dets, int num, float thresh, char **names, image **alphabet, int classes, int frameNumber)
{

	BlobsByFrame* blobs = new BlobsByFrame();
	//instantiate detectors/matchers

	for (int i = 0; i < num; ++i)
	{
		std::vector<std::pair< float, std::string >> _classes;

		int _class = -1;
		float prob = 0.0f;
		// set the class to this blobs
		for (int j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j] > thresh)
			{
				prob = dets[i].prob[j] * 100;

				_classes.push_back(std::make_pair(dets[i].prob[j] * 100, names[j] ));
				_class = j;
			}
		}

		if (_classes.size() > 1)
		{
			// Using simple sort() function to sort 
			sort(_classes.begin(), _classes.end());

			reverse(_classes.begin(), _classes.end());
			
		}

		float verticalScale = (1.0f*input.cols) / input.rows;
		// a class was found
		if (_class >= 0)
		{
			Blob* b = Blob::getInstance(frameNumber);
			b->color = cv::Scalar(((i + 15) * 314) % 255, ((i + 31) * 233) % 255, ((i + 23) * 155) % 255);
			b->guid = randomString();
			b->classes.swap(_classes);

			box f = dets[i].bbox;
			f.y =  (f.y - 0.5)*  verticalScale + 0.5;
			f.h *= verticalScale;
			cv::Rect r((f.x - f.w / 2) * input.cols, (f.y - f.h / 2) * input.rows, f.w* input.cols, f.h* input.rows);
			b->topLeft = cv::Point(f.x * input.cols, f.y * input.rows);
			r.x = min(max(r.x, 0), input.cols - 1);
			r.y = min(max(r.y, 0), input.rows - 1);

			r.width = min(r.width, input.cols - r.x);
			r.height = min(r.height, input.rows - r.y);

			b->updatePos(r, cv::Size(input.cols, input.rows));
			b->assigned = 1;
			blobs->trackBlobs.push_back(b);
		}
	}


	blobs->enqueue(input, input, "", 0, 10);

	return blobs;
}

void removeOverlapped(std::vector<Blob*>& blobs)
{

	for (int i = 0; i < blobs.size(); i++)
	{
		for (int j = 0; j < blobs.size(); j++)
		{
			if (i == j) continue;
			if (blobs[i] == NULL) continue;
			if (blobs[j] == NULL) continue;

			double as = areaSuperposed(blobs[i]->getRect(), blobs[j]->getRect());

			if (as > 0.8)
			{
				blobs[j] = NULL;
			}
		}
	}

	std::vector<Blob*> temp;

	for (int i = 0; i < blobs.size(); i++)
	{
		if (blobs[i])
		{
			temp.push_back(blobs[i]);
		}
	}

	temp.swap(blobs);
}



trackingLib::BlobsByFrame* detectOnFrameCNN(network *net, cv::Mat& mM, int nframe, bool draw)
{

	float* frameData = NULL; // = frameToCNNImage(net, mM);
	//Resize for network
	int key = 0;
	net->input = frameData;
	net->truth = 0;
	net->train = 0;
	net->delta = 0;

	// predict
	int origLayers = net->n;
	int lIndex = 3;

	network_state state = { 0 };
	state.index = 0;
	state.net = *net;
	state.delta = 0;
	state.train = 0;
	state.input = net->input;

	YoloCL::predictYoloCL(net,state, 1, 0);


	startProcess((char*)"converToBlobs");
	// read results
	int nboxes = 0;
	layer l = net->layers[net->n - 1];
	detection* dets = get_network_boxes(net, mM.cols, mM.rows, 0.5, 0.5, 0, 1, &nboxes,0);
	float nms = 0.45f;
	if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

	
	/// Convert the blobs to our format
	BlobsByFrame* bFs = _converToBlobs(mM, dets, nboxes, 0.45, _names, _alphabet, l.classes, nframe);
	removeOverlapped(bFs->trackBlobs);

	std::cout << "potential blobs" << bFs->trackBlobs.size() << "\n";
	endProcess((char*)"converToBlobs");
	return bFs;
}
#endif



float get_colorX(int c, int x, int max)
{

	float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

	float ratio = ((float)x / max) * 5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
	//printf("%f\n", r);
	return r;
}

void cl_draw_detections(cv::Mat& im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
	int i, j;

	for (i = 0; i < num; ++i)
	{
		char labelstr[4096] = { 0 };
		int _class = -1;
		float prob = 0.0f;
		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j] > thresh)
			{
				if (_class < 0) {
					strcat(labelstr, names[j]);
					_class = j;
				}
				else {
					strcat(labelstr, ", ");
					strcat(labelstr, names[j]);
				}

				prob = dets[i].prob[j] * 100;

			}
		}
		if (_class >= 0)
		{
			int width = im.rows * .006;


			//printf("%d %s: %.0f%%\n", i, names[class], prob*100);
			int offset = _class * 123457 % classes;
			float red = get_colorX(2, offset, classes);
			float green = get_colorX(1, offset, classes);
			float blue = get_colorX(0, offset, classes);
			float rgb[3];

			//width = prob*20+2;

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = dets[i].bbox;


			int left = (b.x - b.w / 2.)*im.cols;
			int right = (b.x + b.w / 2.)*im.cols;
			int top = (b.y - b.h / 2.)*im.rows;
			int bot = (b.y + b.h / 2.)*im.rows;

			if (left < 0) left = 0;
			if (right > im.cols - 1) right = im.cols - 1;
			if (top < 0) top = 0;
			if (bot > im.rows - 1) bot = im.rows - 1;

			std::string labels(labelstr);
			labels += std::to_string(prob);

			cv::rectangle(im, cv::Rect(left, top, right - left, bot - top), cv::Scalar(red * 255, green * 255, blue * 255), 2);
			cv::putText(im, labels, cv::Point(left, top), 1, 1.2, cv::Scalar(red * 255, green * 255, blue * 255));


		}
	}

}


void cnnInstance::drawResults(cv::Mat& m)
{
	layer l = net->layers[net->n - 1];

	int _nboxes = 0;
	
	detection* _dets = get_network_boxes(net, im.w, this->im.h, thresh, hier_thresh, 0, 1, &_nboxes, 0);
	if (nms) do_nms_sort(_dets, _nboxes, l.classes, nms);

	if (_dets)
	{
		cl_draw_detections(m, _dets, _nboxes, thresh, this->names, NULL, l.classes);
	}
	delete(_dets);
}


#ifdef TRACKING_LIB
trackingLib::BlobsByFrame* cnnInstance::readResults(bool doRemoveOverlap, float threshold)
{
	auto oldDets = dets;
	int oldBoxes = nboxes;
	layer l = net->layers[net->n - 1];
	dets = get_network_boxes(net, im.w, this->im.h, threshold, threshold, 0, 1, &this->nboxes, 0);
	if (nms) do_nms_sort(this->dets, this->nboxes, l.classes, nms);

	if (oldDets)
	{
		free_detections(oldDets, oldBoxes);
	}
	// Convert to blobs
	trackingLib::BlobsByFrame* bFs = _converToBlobs(mM, dets, nboxes, threshold, this->names, NULL, l.classes, nframe);
	if (doRemoveOverlap)
	{
		removeOverlapped(bFs->trackBlobs);
	}
	return bFs;

}
#endif

cnnInstance::cnnInstance(network* n, size_t _id)
{
	this->net = n;
	this->dets = NULL;
	this->nframe = 0;
	this->imData = NULL;
	this->imSizeData = NULL;

	this->id = _id;
}

void cnnInstance::setFrame(cv::Mat& frame)
{
	mM = frame.clone();
	frame.copyTo(mM);
	updateInput();
}

cv::Mat cnnInstance::GetSquareImage(cv::Mat& squared, cv::Mat& img, int target_width)
{
	int width = img.cols,
		height = img.rows;

	if (squared.cols == 0)
	{
		squared = cv::Mat::zeros(target_width, target_width, img.type());
	}

	int max_dim = (width >= height) ? width : height;
	float scale = ((float)target_width) / max_dim;
	cv::Rect roi;
	if (width >= height)
	{
		roi.width = target_width;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = (target_width - roi.height) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = target_width;
		roi.width = width * scale;
		roi.x = (target_width - roi.width) / 2;
	}

	cv::resize(img, squared(roi), roi.size());

	return squared;
}


void cnnInstance::updateInput()
{

	//Take image data
	if (!im.data)
	{
		im = make_imageX(this->mM.cols, this->mM.rows, 3);
		imData = (float*)calloc(net->w * net->h * net->c, sizeof(float));
		sized = make_image(net->w, net->h, 3);
	}

	this->GetSquareImage(squared, this->mM, net->w);

	//Resize for network
	int key = 0;
	mat_to_imageX(squared, sized);
//	imshow("squared", squared);

	net->input = sized.data;
	net->truth = 0;
	net->train = 0;
	net->delta = 0;
}


void inner_forward_network(network net, network_state state)
{
	state.workspace = net.workspace;
	int i;
	for (i = 0; i < net.n; ++i) {
		state.index = i;
		layer l = net.layers[i];
		
		l.forward(l, state);
		//printf("%d - Predicted in %lf milli-seconds.\n", i, ((double)get_time_point() - time) / 1000);
		state.input = l.output;

		YoloCL::computAndPrint(l,i);

	}
}


void cnnInstance::predict(bool useBackGroundSub)
{
	network_state state = { 0 };
	state.index = 0;
	state.net = *net;
	state.delta = 0;
	state.train = 0;
	state.input = net->input;


	if (gpu_index < 0)
	{
		inner_forward_network(*net, state);
	}
	else
	{
			YoloCL::predictYoloCL(net, state, nframe, yoloMode, useBackGroundSub);
	}
}