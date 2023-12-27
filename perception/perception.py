from pydrake.all import RigidTransform
from pydrake.geometry import Rgba
from pydrake.perception import PointCloud


class CustomPointCloud:
    def __init__(self, xyzs, normals):
        self._xyzs = xyzs
        self._normals = normals
        self.n = self._xyzs.shape[1]

    def xyzs(self):
        return self._xyzs

    def xyz(self, i):
        return self._xyzs[:, i]

    def normal(self, i):
        return self._normals[:, i]

    def normals(self):
        return self._normals

    def transformed(self, X_WC: RigidTransform):
        return CustomPointCloud(
            xyzs= X_WC @ self._xyzs,
            normals=X_WC.rotation() @ self._normals
        )

    def size(self):
        return self.n

    def visualize(self, name, meshcat):
        pc = PointCloud(self.n)
        pc.mutable_xyzs()[:] = self._xyzs
        meshcat.SetObject(
            name, pc, rgba=Rgba(1.0, 0, 0), point_size=0.01
        )
