
import numpy as np
import cv2 

class TransFormation:

    def __init__(self, img_a, img_b, a_pts, b_pts):

        self.reset(img_a, img_b, a_pts, b_pts)

    def reset(self, img_a, img_b, a_pts, b_pts):

        self.img_a = img_a
        self.img_b = img_b
        self.a_pts = a_pts 
        self.b_pts = b_pts 
        d, Z_pts, Tform = self.procrustes(a_pts, b_pts)
        self.Tform = Tform
        self.R = np.eye(3)
        self.R[0:2, 0:2] = self.Tform['rotation']
        self.S = np.eye(3) * self.Tform['scale']
        self.S[2,2] = 1
        self.t = np.eye(3)
        self.t[0:2,2] = self.Tform['translation']
        self.M = np.dot(np.dot(self.R,self.S), self.t.T).T 
        self.height = img_a.shape[0]
        self.width = img_a.shape[1]

        return cv2.warpAffine(self.img_b, self.M[0:2,:], (self.height, self.width))

    def procrustes(self, X, Y, scaling=True, reflection='best'):
        """
        A port of MATLAB's `procrustes` function to Numpy.

        Procrustes analysis determines a linear transformation (translation,
        reflection, orthogonal rotation and scaling) of the points in Y to best
        conform them to the points in matrix X, using the sum of squared errors
        as the goodness of fit criterion.

            d, Z, [tform] = procrustes(X, Y)

        Inputs:
        ------------
        X, Y    
            matrices of target and input coordinates. they must have equal
            numbers of  points (rows), but Y may have fewer dimensions
            (columns) than X.

        scaling 
            if False, the scaling component of the transformation is forced
            to 1

        reflection
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

        Outputs
        ------------
        d       
            the residual sum of squared errors, normalized according to a
            measure of the scale of X, ((X - X.mean(0))**2).sum()

        Z
            the matrix of transformed Y-values

        tform   
            a dict specifying the rotation, translation and scaling that
            maps X --> Y

        """

        n,m = X.shape
        ny,my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection != 'best':

            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0

            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if scaling:

            # optimum scaling of Y
            b = traceTA * normX / normY

            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2
            # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX

        # transformation matrix
        if my < m:
            T = T[:my,:]
        c = muX - b*np.dot(muY, T)
        #rot =1
        #scale=2
        #translate=3
        #transformation values 
        tform = {'rotation':T, 'scale':b, 'translation':c}

        return d, Z, tform

    def transform(self, img):
        return cv2.warpAffine(img, self.M[0:2,:], (self.height, self.width))

    def info(self):
        info = {
            'rows' : f'{self.height}',
            'columns' : f'{self.width}',
            'reference_img': self.img_a,
            'transformed_img': self.img_b,
            'Transform_values': self.Tform
        }

        return info 

# def transform(img_a, img_b, a_pts, b_pts):
#     d, Z_pts, Tform = procrustes(a_pts, b_pts)
#     R = np.eye(3)
#     R[0:2, 0:2] = Tform['rotation']
#     S = np.eye(3) * Tform['scale']
#     S[2,2] = 1
#     t = np.eye(3)
#     t[0:2,2] = Tform['translation']
#     M = np.dot(np.dot(R,S), t.T).T 
#     height = img_a.shape[0]
#     width = img_a.shape[1]
#     return cv2.warpAffine(img_b, M[0:2,:], (height, width))