import vpython as vp
import numpy as np
from utils import generate_basis_combinations


class CrystalView:
    def __init__(self, crystal):
        self.bravais_lattice = crystal.bravais_lattice
        self.motif = crystal.motif
        self.ndim = crystal.ndim
        self.atoms = []
        self.extra_atoms = []
        self.bonds = []
        self.extra_bonds = []
        try:
            self.neighbours = crystal.neighbours
        except AttributeError:
            pass

        vp.scene.visible = False
        vp.scene.background = vp.color.white
        self.create_atoms()
        self.create_bonds()

        # Canvas attributes
        self.drag = False
        self.initial_position = None
        self.cell_vectors = None
        self.cell_boundary = None

    def create_atoms(self):
        # Origin cell
        mesh_points = generate_basis_combinations(self.ndim)
        color_list = [vp.color.yellow, vp.color.red]
        for position in self.motif:
            species = int(position[3])
            atom = vp.sphere(radius=0.1, color=color_list[species])
            atom.pos.x, atom.pos.y, atom.pos.z = position[:3]
            self.atoms.append(atom)

        # Super cell
        if self.bravais_lattice is not None:
            for point in mesh_points:
                unit_cell = np.array([0., 0., 0.])
                if np.linalg.norm(point) == 0:
                    continue
                for n in range(self.ndim):
                    unit_cell += point[n] * self.bravais_lattice[n]
                for position in self.motif:
                    species = int(position[3])
                    atom = vp.sphere(radius=0.1, color=color_list[species])
                    atom.visible = False
                    atom.pos.x, atom.pos.y, atom.pos.z = (position[:3] + unit_cell)
                    self.extra_atoms.append(atom)

    def create_bonds(self):
        mesh_points = generate_basis_combinations(self.ndim)
        try:
            # Origin cell
            for i, neighbours in enumerate(self.neighbours):
                for neighbour in neighbours:
                    unit_cell = vp.vector(neighbour[1][0], neighbour[1][1], neighbour[1][2])
                    bond = vp.curve(self.atoms[i].pos, self.atoms[neighbour[0]].pos + unit_cell)
                    self.bonds.append(bond)
            # Super cell
            for point in mesh_points:
                unit_cell = np.array(([0., 0., 0.]))
                for n in range(self.ndim):
                    unit_cell += point[n] * self.bravais_lattice[n]
                unit_cell = vp.vector(unit_cell[0], unit_cell[1], unit_cell[2])
                for bond in self.bonds:
                    supercell_bond = vp.curve(bond.point(0)["pos"] + unit_cell, bond.point(1)["pos"] + unit_cell)
                    supercell_bond.visible = False
                    self.extra_bonds.append(supercell_bond)

        except AttributeError:
            print("Warning: To visualize bonds, Viewer must receive an initialized System")

    def visualize(self):
        vp.scene.visible = True
        vp.button(text="supercell", bind=self.__add_unit_cells)
        vp.button(text="primitive unit cell", bind=self.__remove_unit_cells)
        vp.button(text="remove bonds", bind=self.__remove_bonds)
        vp.button(text="show bonds", bind=self.__show_bonds)
        vp.button(text="draw cell boundary", bind=self.__draw_boundary)
        vp.button(text="remove cell boundary", bind=self.__remove_boundary)
        vp.scene.bind("mousedown", self.__mousedown)
        vp.scene.bind("mousemove", self.__mousemove)
        vp.scene.bind("mouseup", self.__mouseup)

    def __mousedown(self):
        self.initial_position = vp.scene.mouse.pos
        self.drag = True

    def __mousemove(self):
        if self.drag:
            increment = vp.scene.mouse.pos - self.initial_position
            self.initial_position = vp.scene.mouse.pos
            vp.scene.camera.pos -= increment

    def __mouseup(self):
        self.drag = False

    def __add_unit_cells(self):
        for atom in self.extra_atoms:
            atom.visible = True

    def __show_bonds(self):
        for bond in self.bonds:
            bond.visible = True
        if self.extra_atoms[0].visible:
            for bond in self.extra_bonds:
                bond.visible = True

    def __remove_unit_cells(self):
        for atom in self.extra_atoms:
            atom.visible = False
        if self.extra_bonds[0].visible:
            for bond in self.extra_bonds:
                bond.visible = False

    def __remove_bonds(self):
        for bond in self.bonds:
            bond.visible = False
        for bond in self.extra_bonds:
            bond.visible = False

    def __draw_boundary(self):
        cell_vectors = []
        mid_point = np.array([0., 0., 0.])
        size_vector = [0., 0., 0.]
        for n in range(self.ndim):
            mid_point += self.bravais_lattice[n] / 2
            size_vector[n] = self.bravais_lattice[n][n]
        self.cell_boundary = vp.box(pos=vp.vector(mid_point[0], mid_point[1], mid_point[2]),
                                    size=vp.vector(size_vector[0], size_vector[1], size_vector[2]),
                                    opacity=0.5)
        for basis_vector in self.bravais_lattice:
            unit_cell = vp.vector(basis_vector[0], basis_vector[1], basis_vector[2])
            vector = vp.arrow(pos=vp.vector(0, 0, 0),
                              axis=unit_cell,
                              color=vp.color.green,
                              shaftwidth=0.1)
            cell_vectors.append(vector)

    def __remove_boundary(self):
        self.cell_boundary.visible = False
        for vector in self.cell_vectors:
            vector.visible = False
