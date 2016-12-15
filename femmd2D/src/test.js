// prime.js


function rotate(v, angle) {
    var degrees = 180 / Math.PI;
    angle = angle/degrees
    var nx = (v.x * Math.cos(angle)) - (v.y * Math.sin(angle));
    var ny = (v.x * Math.sin(angle)) + (v.y * Math.cos(angle));

    v.x = nx;
    v.y = ny;

    return v;
};

function initParticles(ucells, L, v0){
	var c = {x :0.0, y: 0.0};
	var gap = {x : 0.0, y: 0.0};
	var n, nx, ny;
	gap.x = L / ucells;
	gap.y = L / ucells;
	n = 0;
	for (ny = 0; ny < ucells; ny++) {
		for (nx = 0; nx < ucells; nx++) {
			c.x = nx + 0.5;
			c.y = ny + 0.5;
			c.x = c.x*gap.x;
			c.y = c.y*gap.y;
			System.add_particle(c.x, c.y, 0.0, 0.0, 1.0, 1.0, 0);
			++n;
		}
	}	
	System.set_velocities(v0);
}

function get_square_nodes(N, Lx, v0, fccFlag) {
    var nodes = read_file("squareVertices.txt", FILECONTENT.FLOAT);
    var cells = read_file("squareTriangles.txt", FILECONTENT.INT);
    var area = 1;
    var Ndisk = nodes.length / 2;
    var nu = 2.0  // ad hoc packing fraction, max packing fraction for disks is 0.9069
    //radius of particles inside elastomer
    var r0 = 0.5 * Math.sqrt(nu * area/ (Ndisk*Math.PI)); 
    var scale = 0.5 / r0;
    //scale= 1.0 / r0;
    System.set_scale(scale);
    print("Elastomer params ", Ndisk, r0, scale);
    //scale simulation box
    System.set_box(scale*Lx,scale*Lx);
    //scale particles pos
    var i;
    for (i = 0; i < nodes.length; i++){
        nodes[i] = nodes[i]*scale;
    }
}

function add_elastomers1(N, Lx, v0, fccFlag) {
}


//addElastomer(nodes,cells,x,y,typ,g,sig,mass,exc);
function add_elastomers(N, Lx, v0, fccFlag) {


    var nodes = read_file("squareVertices5v.txt", FILECONTENT.FLOAT);
    var cells = read_file("squareTriangles5v.txt", FILECONTENT.INT);
    var do_rotation = false;
    //var nodes = read_file("triangleVertices1.txt", FILECONTENT.FLOAT);
    //var cells = read_file("triangleTriangles1.txt", FILECONTENT.INT);
    //var txt = readFile("Makefile", FILECONTENT.TEXT);
    //print(txt);

	// Calculate scale factor s = 1/r0 = 1/(V0/(N*0.9*PI))
	// r0 radius of inner particles whet the elastomer disk has Radius 1
    // V0 = PI*R^2
    var area = 1;
    var Ndisk = nodes.length / 2;
    var nu = 0.91  // ad hoc packing fraction, max packing fraction for disks is 0.9069
    //radius of particles inside elastomer
    //var r0 = 0.5*Math.sqrt(nu / Ndisk); // 0.75 ad hoc factor
    var r0 = 0.5 * Math.sqrt(nu * area/ (Ndisk*Math.PI));
    //scale calculater from
    // (0.5*s+0.5) = (s*r0+0.5) check in office for cacls
    var scale = 1.0 / r0;
    System.set_scale(scale);
    print("Elastomer params ", Ndisk, r0, scale);
	//printf("r0 %f scale %f R %f A0 %f Af %f expected A %f\n", r0, scale, scale*r0, PI*r0*r0, PI*r0*r0*scale*scale, PI*0.5*0.5);
	//scale simulation box
	System.set_box(scale*Lx,scale*Lx);
	//scale particles pos
	var i;
	for (i = 0; i < nodes.length; i++){
		nodes[i] = nodes[i]*scale;
	}
	var R = scale;
	var L = scale * Lx;
	print("Radius ", scale);
	System.set_range_rdf(5 * R);
	var deltaR = 0.0 * R;
	var rotNodes = []
	for (i = 0; i < Ndisk; i++) {
	    var vn = { x: nodes[2 * i], y: nodes[2 * i + 1] };
	    var v = vn;
        if(do_rotation)
            v = rotate(vn, 180);
	    rotNodes[2 * i] = v.x;
	    rotNodes[2 * i + 1] = v.y - deltaR;
	}
	var rotatedNodes = new Float32Array(rotNodes);
	rototeNodes = nodes;
	
	var c = {x :0.0, y: 0.0};
	var nx, ny;
	var fac1 = R;
	var fac2 = fac1 / 3;
	fac2 = 0;
	n = 0;
	if (fccFlag == 0) {
	    var ncells = Math.sqrt(N / 2);
	    var b = L / Math.sqrt(N);
	    for (ny = 0; ny < ncells; ny++) {
	        for (nx = 0; nx < ncells; nx++) {
	            c.x = nx + 0.5;
	            c.y = ny + 0.5;
	            c.x = c.x * b + fac2;
	            c.y = c.y * b + fac2;
	            System.add_elastomer(nodes, cells, c.x, c.y, 0, 0, 1.0, N, true);
	            ++n;
	        }
	    }
	} else {
	    var ncells = Math.floor(Math.sqrt(N / 2));
	    var b = L / ncells
	    print("Elastomer lattice ", b, ncells, L);
	    //var b = L / Math.sqrt(N);
	    //b = L / 2;
	    //var ncells = (L / b);
	    //ncells = 2;
	    a1 = { x: 0.0, y: 0.0 };
	    a2 = { x: 0.5, y: 0.5 };
	    var c1 = { x: 0.0, y: 0.0 };
	    var c2 = { x: 0.0, y: 0.0 };
	    for (ny = 0; ny < ncells; ny++) {
	        for (nx = 0; nx < ncells; nx++) {
	            c1.x = nx + a1.x;
	            c1.y = ny + a1.y;
	            c2.x = nx + a2.x;
	            c2.y = ny + a2.y;
	            c.x = c1.x * b + fac2;
	            c.y = c1.y * b + fac2;
	            System.add_elastomer(nodes, cells, c.x, c.y, 0, 0, 1.0, N, true);
	            c.x = c2.x * b + fac2;
	            c.y = c2.y * b + fac2;
	            System.add_elastomer(rotatedNodes, cells, c.x, c.y, 0, 0, 1.0, N, true);
	            ++n;
	        }
	    }
	}
	System.set_velocities(v0);
}


function runOne(density, yConstant) {
    var ucells = 8;
	var N = 2*ucells*ucells;
	var dens = density;
	var L = Math.sqrt(N / dens);
	var T = 1.0;
	var v0 = Math.sqrt(2.0 * T * (1.0 - 1.0 / N));
	print("box size ", density, L, L);
	System.init(L,L);
	System.set_dt(0.001);
	System.set_equilibrium_steps(10000);
	//System.set_kBond(10);
    //System.set_kArea(0.0);
    var E = yConstant
    System.set_youngs_modulus(E);
    

	var eps = new Float64Array([1.0]);
	//var rc = new Float64Array([2.6]);
	var rc = new Float64Array([Math.pow(2.0,1.0/6.0)]);
	var us = new Float64Array([1.0]);
	System.set_pair_potential(eps, rc, us);
	add_elastomers(N, L, v0, 1);
    //initParticles(ucells,L,v0);
	print("system info");
	var info = System.info();
	//str = JSON.stringify(info);
	str = JSON.stringify(info,null, 4);
	print(str);
	
	System.allocate_cuda();
	System.init_neighbor_list();
	dir = "data_rho_" + dens.toFixed(2).toString() + "_E_" + E.toFixed(2).toString();
	print(dir);
	make_dir(dir);
	change_dir(dir);
	System.save_trajectory(true);
	System.set_trajectory_steps(100);
    System.save_vtk("intial_state");
	System.save_vtf("movie.vtf", 0);
	System.calc_cluster_props();
	System.save_cluster_vtf("movieCluster.vtf", 0);
	System.save_cluster_xyz("movie.xyz", 0);
	var i;

	var props = System.get_props();
	str = JSON.stringify(props, null, 4);
    //print(str)
	str = props.steps.toString() + " " + props.uKinetic.avgval.toString() + "\n";
	write_to_file("daata.dat", str);

	print("starting run");
	for(i = 0; i < 200; i++ ){
	    System.run(5000);
        print("\n")
        System.print_props();
        var props = System.get_props();
        str = JSON.stringify(props, null, 4);
        //print(str);
        str = props.steps.toString() + " " + props.uKinetic.avgval.toString() + "\n";
        append_to_file("data.dat", str);
		System.save_vtk("movie");
		System.save_vtf("movie.vtf", 1);
		System.save_cluster_vtf("movieCluster.vtf", 1);
		System.save_cluster_xyz("movie.xyz", 1);
	}
}

function mdmain() {
    print("running mdmain");
    System.DEV(DEV.CPU);
    var E = 100;
    //E = [2.0, 5.0, 10.0, 20.0, 50, 100]
    E = [10.0, 20.0, 50.0];
    E = [50.0]
    var rho = [0.4];
    //rho = [0.5,0.6,0.7,0.8,0.9]
    //var pf = 0.6;
    //var rho = 8 * pf / (6* Math.sqrt(3.0) * 0.5 * 0.5);
    //print(rho);
    for (var i = 0; i < E.length ; i++) {
        for (var j = 0; j < rho.length; j++) {
            change_dir("C:\\tmp");
            //change_dir("/home/manuel/tmp");
            runOne(rho[j], E[i]);
            System.reset();
        }
    }
}
