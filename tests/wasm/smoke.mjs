// Smoke test for @meshioplusplus/wasm, run in CI (.github/workflows/wasm.yml)
// after every build and usable as a live usage example. Round-trips a
// synthetic mesh through 3 representative formats (VTU binary+zlib, STL
// binary, OBJ ascii) plus a plain-text read, and exercises writeMesh/
// readMesh/convert/numNodesPerCell through the package's own public API
// (src/index.mjs) -- not the raw embind glue -- so this is exactly what a
// real consumer would call.
//
// The package ships two native artifacts: the sequential meshioplusplus_wasm
// and the threaded (OpenMP/pthreads) meshioplusplus_wasm_mt. The full suite
// below runs against the THREADED build (forced with { variant: 'mt' }) so the
// parallel code paths are what CI exercises -- Wasm threads work under Node
// with no cross-origin-isolation headers. A compact sanity block at the very
// end loads the sequential build too, so both artifacts are proven loadable.
//
// Usage: node tests/wasm/smoke.mjs   (after `build/configure-wasm.sh --build`
// has populated src/wasm/dist/meshioplusplus_wasm{,_mt}.{mjs,wasm})

import assert from 'node:assert/strict';
import { loadMeshioPlusPlus } from '../../src/wasm/src/index.mjs';

let failed = false;
function step(name, fn) {
    try {
        fn();
        console.log(`ok - ${name}`);
    } catch (err) {
        failed = true;
        console.error(`NOT OK - ${name}`);
        console.error(err);
    }
}

const m = await loadMeshioPlusPlus({}, { variant: 'mt' });
step('threaded (mt) build reports the openmp parallel backend', () => {
    // The whole point of the mt artifact: it must actually be the OpenMP build,
    // not a mislabelled sequential one. parallelBackend() is exposed by the
    // embind binding; a build configured with SEQ would report "seq" here.
    assert.equal(typeof m.parallelBackend, 'function');
    assert.equal(m.parallelBackend(), 'openmp');
});

// A small synthetic tetrahedron + a point/cell data field, built directly as
// a JS mesh object (bypassing file I/O) to test the writeMesh(object) path.
const tet = {
    points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    dim: 3,
    cells: [{ type: 'tetra', data: new Int32Array([0, 1, 2, 3]), nodesPerCell: 4 }],
    point_data: { temperature: new Float64Array([1, 2, 3, 4]) },
    cell_data: { material: [new Float64Array([7])] },
    field_data: {},
};

step('numNodesPerCell metadata table', () => {
    const table = m.numNodesPerCell();
    assert.equal(table.tetra, 4);
    assert.equal(table.triangle, 3);
});

step('topologicalDimension metadata table', () => {
    const table = m.topologicalDimension();
    assert.equal(table.tetra, 3);
    assert.equal(table.triangle, 2);
});

step('VTU binary+zlib round-trip (object -> file -> object)', () => {
    m.writeMesh('/tet.vtu', tet);
    const back = m.readMesh('/tet.vtu');
    assert.equal(back.points.length, 12);
    assert.equal(back.cells.length, 1);
    assert.equal(back.cells[0].type, 'tetra');
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2, 3]);
    assert.deepEqual(Array.from(back.point_data.temperature), [1, 2, 3, 4]);
    assert.deepEqual(Array.from(back.cell_data.material[0]), [7]);
});

step('STL binary round-trip', () => {
    // STL is a surface-only format (one triangle soup, no volume cells) --
    // writing a "tetra" block to it is a legitimate no-op (matches native
    // meshio++: `mesh.write("x.stl")` on a tetra-only mesh silently produces
    // an empty "solid\nendsolid\n", with a warning), so this needs its own
    // triangle-only mesh rather than reusing `tet`.
    const tri = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        dim: 3,
        cells: [{ type: 'triangle', data: new Int32Array([0, 1, 2]), nodesPerCell: 3 }],
    };
    m.writeMesh('/tri.stl', tri, 'stl');
    const back = m.readMesh('/tri.stl', 'stl');
    assert.equal(back.cells.length, 1);
    assert.equal(back.cells[0].type, 'triangle');
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2]);
});

step('OBJ ascii read from a hand-written file', () => {
    const obj = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n';
    m.FS.writeFile('/tri.obj', obj);
    const mesh = m.readMesh('/tri.obj');
    assert.equal(mesh.points.length, 9);
    assert.equal(mesh.cells.length, 1);
    assert.equal(mesh.cells[0].type, 'triangle');
    assert.deepEqual(Array.from(mesh.cells[0].data), [0, 1, 2]);
});

step('convert() reads one format and writes another directly', () => {
    m.convert('/tri.obj', '/tri.vtk');
    const back = m.readMesh('/tri.vtk');
    assert.equal(back.cells[0].type, 'triangle');
});

step('format collision: .msh defaults to gmsh, explicit override selects ansys', () => {
    m.writeMesh('/tet.msh', tet, 'ansys');
    let threw = false;
    try {
        m.readMesh('/tet.msh', 'gmsh'); // an ansys file is not valid gmsh
    } catch (err) {
        threw = true;
        assert.ok(err instanceof Error);
        assert.ok(err.message.length > 0, 'error should carry a real message');
    }
    assert.ok(threw, 'expected reading an ansys-written .msh as gmsh to throw');
});

step('unknown format raises a catchable Error, not an abort', () => {
    assert.throws(() => m.readMesh('/does/not/exist.obj'), /Could not open file/);
});

// A triangle mesh reused across the extra format round-trips below.
const tri2 = {
    points: new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
    dim: 3,
    cells: [{ type: 'triangle', data: new Int32Array([0, 1, 2, 0, 2, 3]), nodesPerCell: 3 }],
};

step('PLY binary round-trip', () => {
    m.writeMesh('/tri.ply', tri2, 'ply');
    const back = m.readMesh('/tri.ply', 'ply');
    assert.equal(back.cells.length, 1);
    assert.equal(back.cells[0].type, 'triangle');
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2, 0, 2, 3]);
});

step('OFF ascii round-trip', () => {
    m.writeMesh('/tri.off', tri2, 'off');
    const back = m.readMesh('/tri.off', 'off');
    assert.equal(back.cells[0].type, 'triangle');
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2, 0, 2, 3]);
});

step('GMSH ascii round-trip (tetra, volume format)', () => {
    m.writeMesh('/tet.gmsh.msh', tet, 'gmsh');
    const back = m.readMesh('/tet.gmsh.msh', 'gmsh');
    assert.equal(back.cells[0].type, 'tetra');
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2, 3]);
    assert.equal(back.points.length, 12);
});

// --------------------------------------------------------------------------
// HDF5- and netCDF-backed formats. These exist only because
// build/build-wasm-deps.sh produced a wasm32 libhdf5/libnetcdf for this build,
// so they are round-tripped rather than read from a fixture: writing exercises
// the deflate filter (every HDF5 writer here compresses at gzip level 4), and
// a fixture would need a Git-LFS reference file for something the build can
// generate itself.
// --------------------------------------------------------------------------

// MED is the one exception to writing `tet` as-is: the C++ MED writer defers a
// mesh carrying named fields to the Python reference writer (CHA fields with
// MED-4.1 bitmask/units/step metadata), and there is no Python anywhere in a
// wasm build -- so here that documented fallback is simply an unsupported
// case, and the geometry-only mesh is what this build can write. See
// doc/wasm.md and doc/formats/med.md.
const tetNoData = { points: tet.points, dim: 3, cells: tet.cells };

for (const [format, path, mesh] of [
    ['med', '/tet.med', tetNoData],
    ['cgns', '/tet.cgns', tet],
    ['h5m', '/tet.h5m', tet],
    ['hmf', '/tet.hmf', tet],
    ['exodus', '/tet.exo', tet],
]) {
    step(`${format} round-trip (HDF5/netCDF-backed)`, () => {
        m.writeMesh(path, mesh, format);
        // A container that never reached the disk would read back as an
        // unrelated failure below; check the bytes exist first.
        assert.ok(m.FS.stat(path).size > 0, `${path} is empty`);
        const back = m.readMesh(path, format);
        assert.equal(back.points.length, 12);
        assert.equal(back.cells.length, 1);
        assert.equal(back.cells[0].type, 'tetra');
        assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2, 3]);
    });
}

step('exodus SPHERE elements and per-element attributes round-trip', () => {
    // The pair of things a particle/peridynamics mesh is made of, and the
    // surface that actually mattered for
    // https://github.com/loumalouomega/VSCode-MDPA-Preview/issues/63: wasm has
    // no Python fallback, so the C++ reader failing on such a file made Exodus
    // unusable in the browser viewer specifically. One-node SPHERE elements are
    // `vertex` cells, and the radius rides in `cell_data` under `exodus:attr:`.
    //
    // This deliberately writes its own file rather than reading the real
    // PeriLab one at tests/python/meshes/exodus/: that fixture is Git-LFS, and
    // wasm.yml checks out WITHOUT `lfs: true` (only ci.yml sets it), so
    // reaching for it here would hand this test a 130-byte pointer. The
    // NUL-terminated `elem_type` that fixture exists for is a property of the
    // shared C++ reader, and tests/python/test_exodus.py pins it there.
    const spheres = {
        points: [0, 0, 0, 1, 0, 0, 2, 0, 0],
        dim: 3,
        cells: [{ type: 'vertex', data: [0, 1, 2], nodesPerCell: 1 }],
        cell_data: { 'exodus:attr:RADIUS': [[0.5, 0.25, 0.125]] },
    };
    m.writeMesh('/spheres.exo', spheres, 'exodus');
    const back = m.readMesh('/spheres.exo', 'exodus');
    assert.equal(back.cells.length, 1);
    assert.equal(back.cells[0].type, 'vertex');
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2]);
    const radius = back.cell_data['exodus:attr:RADIUS'];
    assert.ok(radius, 'the radius attribute must survive the round-trip');
    assert.deepEqual(Array.from(radius[0]), [0.5, 0.25, 0.125]);
});

step('an ASCII read still works after netCDF/HDF5 has run (stack-size guard)', () => {
    // Regression guard for a silent, global corruption, not a niceties check.
    // HDF5's and netCDF-4's frames overran Emscripten's default 64 KiB stack,
    // which grows down into the static data segment: one Exodus write clobbered
    // libc++'s locale facets and every later `istream >> number` -- so every
    // ASCII reader in the module -- trapped. CMakeLists.txt now links with
    // -sSTACK_SIZE=4MB; this asserts that it still does. Order matters: the
    // exodus round-trip above must have run first.
    m.FS.writeFile('/after-exodus.obj', 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n');
    const back = m.readMesh('/after-exodus.obj', 'obj');
    assert.equal(back.points.length, 9);
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2]);
});

step('MED writes plain point_data/cell_data directly (the single-timestep common case)', () => {
    // Previously this threw unconditionally ("fields handled by Python
    // fallback") on ANY data-carrying mesh -- fatal here, since there is no
    // Python fallback in this build to defer to. Now the C++ writer handles
    // ordinary arrays directly; only the enhanced Python-only conventions
    // (multi-timestep name encoding, units, component names) still defer --
    // and none of those are even expressible through this flat JS API's
    // field_data (Record<string, Float64Array>, numeric only), so a WASM
    // caller can never hit that path at all.
    m.writeMesh('/fields.med', tet, 'med');
    const back = m.readMesh('/fields.med', 'med');
    assert.deepEqual(Array.from(back.point_data.temperature), [1, 2, 3, 4]);
    assert.deepEqual(Array.from(back.cell_data.material[0]), [7]);
});

step('med is an options-aware reader (lenient / timeStep reach it)', () => {
    // v9.9.0 registered MED in `registry_readers_ex()`. That table is what
    // carries ReadOptions to this build at all -- WASM has no Python fallback,
    // so before it a multi-timestep or unit-carrying MED file was simply
    // unreadable here, with no flag that could change that. Asserting through
    // the *wrapper*, not Module.*, per this file's standing rule.
    assert.equal(m.readerSupportsOptions('med'), true);

    // And the options are actually honoured end to end. Our own writer emits
    // the single-step shape, so `lenient` is a no-op on it and `timeStep: 0`
    // is the only in-range step -- which is exactly the point: neither may
    // change a well-formed read, and an out-of-range step must still fail.
    m.writeMesh('/opts.med', tet, 'med');
    const lenient = m.readMeshSelective('/opts.med', { format: 'med', lenient: true });
    assert.deepEqual(Array.from(lenient.point_data.temperature), [1, 2, 3, 4]);
    const first = m.readMeshSelective('/opts.med', { format: 'med', timeStep: 0 });
    assert.deepEqual(Array.from(first.point_data.temperature), [1, 2, 3, 4]);
    assert.throws(
        () => m.readMeshSelective('/opts.med', { format: 'med', timeStep: 5 }),
        /step/i,
    );
});

step('xdmf writes an HDF companion file when HDF5 is available', () => {
    // The registry's xdmf writer default follows the build (registry.cpp): with
    // HDF5 linked in it emits Format="HDF" heavy data beside the XML, matching
    // the Python reference writer. A JS caller therefore has TWO files to pull
    // back out of the virtual FS, not one -- that is the whole point of this
    // assertion.
    m.writeMesh('/tet.xdmf', tet, 'xdmf');
    assert.ok(m.FS.stat('/tet.xdmf').size > 0);
    assert.ok(m.FS.stat('/tet.h5').size > 0, 'expected the .h5 companion beside /tet.xdmf');
    const back = m.readMesh('/tet.xdmf', 'xdmf');
    assert.equal(back.cells[0].type, 'tetra');
    assert.deepEqual(Array.from(back.point_data.temperature), [1, 2, 3, 4]);
});

step('malformed file raises a catchable Error, not a WASM abort', () => {
    m.FS.writeFile('/bad.vtu', '<?xml version="1.0"?><NotVTK></NotVTK>');
    assert.throws(
        () => m.readMesh('/bad.vtu', 'vtu'),
        (err) => err instanceof Error && err.message.length > 0,
    );
});

// --- data operations -------------------------------------------------------
// These act on the mesh's data arrays; the geometry must come through
// untouched. `tetv` carries point_data.temperature and cell_data.material.
//
// The WASM mesh-object shape crosses point_data/cell_data/field_data as flat
// Float64Arrays, with the per-array component count carried alongside in a
// sibling `*_components` object (see js_bindings.cpp's mesh_to_val/val_to_mesh,
// and the dedicated round-trip steps further down). The arrays here are plain
// scalars, so they need no such entry -- an absent name means one component.

const tetv = {
    points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    dim: 3,
    cells: [{ type: 'tetra', data: new Int32Array([0, 1, 2, 3]), nodesPerCell: 4 }],
    point_data: {
        temperature: new Float64Array([1, 2, 3, 4]),
        pressure: new Float64Array([10, 20, 30, 40]),
    },
    cell_data: { material: [new Float64Array([7])] },
    field_data: {},
};

step('dataInfo summarizes every array', () => {
    const arrays = m.dataInfo(tetv);
    assert.equal(arrays.length, 3);
    const t = arrays.find((a) => a.name === 'temperature');
    assert.equal(t.location, 'point_data');
    assert.equal(t.numEntries, 4);
    assert.equal(t.numComponents, 1);
    assert.equal(t.min, 1);
    assert.equal(t.max, 4);
    assert.equal(t.mean, 2.5);
    assert.equal(t.numNan, 0);
    const p = arrays.find((a) => a.name === 'pressure');
    assert.equal(p.min, 10);
    assert.equal(p.max, 40);
});

step('dataIntegrate: cell-measure-weighted total/mean, and per-region', () => {
    const tagged = {
        ...tetv,
        regions: [{ name: 'solid', kind: 'cell', dim: 3, tag: 7, entries: Int32Array.from([0]) }],
    };
    const report = m.dataIntegrate(tagged, ['material']);
    assert.equal(report.length, 1);
    const arr = report[0];
    assert.equal(arr.name, 'material');
    assert.equal(arr.numComponents, 1);
    assert.equal(arr.domain.numCells, 1);
    assert.equal(arr.domain.numSkipped, 0);
    assert.ok(arr.domain.domainMeasurePerComponent[0] > 0);
    assert.ok(
        Math.abs(
            arr.domain.meanPerComponent[0] -
                arr.domain.totalPerComponent[0] / arr.domain.domainMeasurePerComponent[0],
        ) < 1e-9,
    );
    assert.equal(arr.domain.numNanPerComponent[0], 0);
    assert.equal(arr.regions.length, 1);
    assert.equal(arr.regions[0].name, 'solid');
    assert.equal(arr.regions[0].numCells, 1);

    // No array filter means every cell_data array (there's exactly one).
    assert.equal(m.dataIntegrate(tetv).length, 1);

    // A point_data-only name throws, naming the fix.
    assert.throws(() => m.dataIntegrate(tetv, ['temperature']), /point_data_to_cell_data/);
});

step('dataCalc evaluates an expression', () => {
    const out = m.dataCalc(tetv, '2 * temperature + 1', 'point', 'derived', false);
    const derived = out.point_data.derived;
    assert.ok(Math.abs(derived[0] - 3) < 1e-12);
    assert.ok(Math.abs(derived[3] - 9) < 1e-12);
    // Geometry is never modified.
    assert.equal(out.points.length, 12);
    assert.equal(out.cells.length, 1);
});

step('dataCalc rejects an unknown function with a catchable Error', () => {
    assert.throws(
        () => m.dataCalc(tetv, 'log(temperature)', 'point', 'bad', false),
        (err) => err instanceof Error && err.message.length > 0,
    );
});

step('dataDrop / dataKeep / dataRename', () => {
    const dropped = m.dataDrop(tetv, 'point', ['temperature'], false);
    assert.ok(!('temperature' in dropped.point_data));
    assert.ok('pressure' in dropped.point_data);

    const kept = m.dataKeep(tetv, 'point', ['temperature'], false);
    assert.deepEqual(Object.keys(kept.point_data), ['temperature']);

    const renamed = m.dataRename(tetv, 'point', 'temperature', 'T');
    assert.ok('T' in renamed.point_data);
    assert.ok(!('temperature' in renamed.point_data));
});

step('dataPointToCell / dataCellToPoint', () => {
    const toCell = m.dataPointToCell(tetv, ['temperature'], '_c');
    // Single tetra: the mean of {1,2,3,4}.
    assert.ok(Math.abs(toCell.cell_data.temperature_c[0][0] - 2.5) < 1e-12);

    const toPoint = m.dataCellToPoint(tetv, ['material'], 'uniform', '');
    assert.equal(toPoint.point_data.material.length, 4);
    assert.ok(Math.abs(toPoint.point_data.material[0] - 7) < 1e-12);
});

step('dataCondition normalizes to [0, 1]', () => {
    const out = m.dataCondition(
        tetv, 'point', ['temperature'], 'normalize', 0, 1,
        'component', 'ignore', 0, '',
    );
    const t = out.point_data.temperature;
    assert.ok(Math.abs(t[0] - 0) < 1e-12);
    assert.ok(Math.abs(t[3] - 1) < 1e-12);
});


// ---------------------------------------------------------------------------
// Selective reads, file summaries, and the codec build profile
// ---------------------------------------------------------------------------

const seltri = {
    points: new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
    dim: 3,
    cells: [{ type: 'triangle', data: new Int32Array([0, 1, 2, 0, 2, 3]), nodesPerCell: 3 }],
    point_data: {
        u: new Float64Array([1, 2, 3, 4]),
        v: new Float64Array([10, 20, 30, 40]),
    },
};
m.writeMesh('/selective.vtu', seltri);

step('readMeshSelective: no options reads every array', () => {
    const mesh = m.readMeshSelective('/selective.vtu');
    assert.deepEqual(Object.keys(mesh.point_data).sort(), ['u', 'v']);
});

step('readMeshSelective: pointsOnly keeps geometry, drops data', () => {
    const mesh = m.readMeshSelective('/selective.vtu', { pointsOnly: true });
    assert.equal(Object.keys(mesh.point_data).length, 0);
    // pointsOnly narrows data, not topology.
    assert.equal(mesh.points.length, 12);
    assert.equal(mesh.cells[0].data.length, 6);
});

step('readMeshSelective: arrays subset, and [] vs null', () => {
    assert.deepEqual(
        Object.keys(m.readMeshSelective('/selective.vtu', { arrays: ['u'] }).point_data), ['u']);
    // The distinction that motivates std::optional all the way down: an empty
    // list means *no* arrays, null means *every* array.
    assert.equal(
        Object.keys(m.readMeshSelective('/selective.vtu', { arrays: [] }).point_data).length, 0);
    assert.deepEqual(
        Object.keys(m.readMeshSelective('/selective.vtu', { arrays: null }).point_data).sort(),
        ['u', 'v']);
});

step('readMetadata summarizes without loading the arrays', () => {
    const meta = m.readMetadata('/selective.vtu');
    assert.equal(meta.numPoints, 4);
    assert.equal(meta.numCells, 2);
    assert.equal(meta.cellBlocks.length, 1);
    assert.equal(meta.cellBlocks[0].type, 'triangle');
    assert.equal(meta.cellBlocks[0].nodesPerCell, 3);
    assert.deepEqual(meta.pointDataNames.sort(), ['u', 'v']);
    assert.equal(meta.format, 'vtu');
    // vtu has a native metadata path, so this really was cheap...
    assert.equal(meta.fellBackToFullRead, false);
    // Always present, so a caller can read .length without testing the key;
    // empty for a format with no time concept.
    assert.deepEqual(meta.timeValues, []);
    // ...and a native summary never decodes the coordinates, so it reports no
    // bbox rather than a fabricated one at the origin.
    assert.ok(!('bboxMin' in meta));
});

step('readMetadata flags a full-read fallback and can then afford a bbox', () => {
    m.writeMesh('/selective.stl', seltri, 'stl');
    const meta = m.readMetadata('/selective.stl');
    assert.equal(meta.fellBackToFullRead, true);
    assert.ok('bboxMin' in meta && 'bboxMax' in meta);
});

step('readerSupportsOptions reports the native paths', () => {
    assert.equal(m.readerSupportsOptions('vtu'), true);
    assert.equal(m.readerSupportsOptions('stl'), false);
    // Exodus became options-aware in v8.6.0 so `timeStep` has somewhere to go.
    // Before that this was false and the format was not readable here at all.
    assert.equal(m.readerSupportsOptions('exodus'), true);
});

step('exodus reads here at all, and reports its time steps', () => {
    // The regression this guards: the reader used to throw on `qa_records`,
    // which every file SEACAS/Cubit/Sierra writes carries -- and there is no
    // Python fallback in this build to defer to. A file written by meshio++'s
    // own writer carries no qa_records, so this cannot prove that part (the
    // pytest suite's hand-authored fixture does); what it does prove is that
    // the format is reachable and the new time plumbing is wired end to end.
    m.writeMesh('/smoke.e', seltri, 'exodus');
    const mesh = m.readMesh('/smoke.e', 'exodus');
    assert.equal(mesh.cells.length, 1);
    assert.equal(mesh.cells[0].type, 'triangle');

    const meta = m.readMetadata('/smoke.e', 'exodus');
    assert.equal(meta.format, 'exodus');
    // meshio++'s writer emits exactly one (dummy) step.
    assert.equal(meta.timeValues.length, 1);

    // Step 0 and -1 both name that single step; anything else is out of range,
    // and must say so rather than silently handing back step 0.
    for (const timeStep of [0, -1]) {
        const one = m.readMeshSelective('/smoke.e', { format: 'exodus', timeStep });
        assert.equal(one.cells[0].type, 'triangle');
    }
    assert.throws(
        () => m.readMeshSelective('/smoke.e', { format: 'exodus', timeStep: 5 }),
        /out of range/,
    );
});

step('zstd/lz4 are compiled out of the WASM build', () => {
    // Consistent with the HDF5/netCDF-backed formats: no Emscripten port
    // exists, so zlib remains the only block codec here.
    m.writeMesh('/codec.vtu', seltri);
    const text = new TextDecoder().decode(m.FS.readFile('/codec.vtu'));
    assert.ok(!text.includes('vtkZSTDDataCompressor'));
    assert.ok(!text.includes('vtkLZ4DataCompressor'));
});

// The geometry operations below are reached through the package wrapper
// (src/index.mjs), not `Module.*` directly. That is the whole point: an embind
// binding the wrapper does not forward is unreachable from
// loadMeshioPlusPlus(), which is exactly how these ops shipped broken before.

const cube = {
    points: new Float64Array([
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
    ]),
    dim: 3,
    cells: [
        { type: 'hexahedron', data: new Int32Array([0, 1, 2, 3, 4, 5, 6, 7]), nodesPerCell: 8 },
    ],
    point_data: {},
    cell_data: {},
    field_data: {},
};

step('convertCells simplexify: one hexahedron -> 6 tetra', () => {
    const out = m.convertCells(cube, 'simplexify');
    assert.equal(out.cells.length, 1);
    assert.equal(out.cells[0].type, 'tetra');
    assert.equal(out.cells[0].data.length, 6 * 4);
    // The decomposition reuses the parent's own corner nodes.
    assert.equal(out.points.length, 24);
});

step('convertCells elevate/linearize round-trips', () => {
    const up = m.convertCells(tet, 'elevate');
    assert.equal(up.cells[0].type, 'tetra10');
    assert.equal(up.points.length, (4 + 6) * 3);
    const down = m.convertCells(up, 'linearize');
    assert.equal(down.cells[0].type, 'tetra');
    assert.equal(down.points.length, 12);
});

step('convertCells rejects a full-Lagrange elevate target', () => {
    const quad9 = {
        points: new Float64Array(9 * 3),
        dim: 3,
        cells: [
            {
                type: 'quad9',
                data: new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
                nodesPerCell: 9,
            },
        ],
        point_data: {},
        cell_data: {},
        field_data: {},
    };
    assert.throws(() => m.convertCells(quad9, 'elevate'));
});

step('subdivide: one hexahedron -> 6 polyhedral children, one apex point', () => {
    // subdivide has no per-type template table: cell_rings/orient_rings
    // handle a tabulated type and a polyhedron block uniformly. One
    // polyhedral child per face (that face unfaned, plus one new triangle
    // per face edge back to a new interior point) -- automatically
    // conforming, unlike refine.
    const out = m.subdivide(cube);
    assert.equal(out.cells.length, 1);
    assert.equal(out.cells[0].type, 'polyhedron');
    // cellOffsets is per-OUTPUT-cell (one per face) -- 6 children -> 7 entries.
    assert.equal(out.cells[0].cellOffsets.length, 7);
    // Each child: 1 original quad face + 4 new triangles = 5 faces;
    // 6 children x 5 faces = 30 faces -> 31 offsets.
    assert.equal(out.cells[0].faceOffsets.length, 31);
    // One new interior point (the apex) added, nothing pruned.
    assert.equal(out.points.length, cube.points.length + 3);
});

step('subdivide: recordParentIds attaches subdivide:parent_cell', () => {
    const out = m.subdivide(cube, true);
    assert.ok('subdivide:parent_cell' in out.cell_data);
    assert.deepEqual(Array.from(out.cell_data['subdivide:parent_cell'][0]), [0, 0, 0, 0, 0, 0]);
});

step('subdivide: non-3D blocks pass through unchanged', () => {
    const flatTri = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        dim: 3,
        cells: [{ type: 'triangle', data: new Int32Array([0, 1, 2]), nodesPerCell: 3 }],
        point_data: {},
        cell_data: {},
        field_data: {},
    };
    const out = m.subdivide(flatTri);
    assert.equal(out.cells[0].type, 'triangle');
    assert.deepEqual(Array.from(out.cells[0].data), Array.from(flatTri.cells[0].data));
    assert.equal(out.points.length, flatTri.points.length);
});

// Two unit hexahedra sharing one face (x=1 plane) -- the fixture agglomerate
// needs a real (non-identity) merge to exercise.
const twoHexes = {
    points: new Float64Array([
        0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 2, 0, 0, 2, 1, 0,
        2, 1, 1, 2, 0, 1,
    ]),
    dim: 3,
    cells: [
        {
            type: 'hexahedron',
            data: new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11]),
            nodesPerCell: 8,
        },
    ],
    point_data: {},
    cell_data: {},
    field_data: {},
};

step('agglomerate: two adjacent hexes merge into one polyhedron', () => {
    const out = m.agglomerate(twoHexes, 2);
    assert.equal(out.cells.length, 1);
    assert.equal(out.cells[0].type, 'polyhedron');
    // cellOffsets is per-OUTPUT-cell -- one merged cell -> 2 entries.
    assert.equal(out.cells[0].cellOffsets.length, 2);
    // 12 total face-references (6 per hex) minus the 2 references to the 1
    // shared, now-internal face.
    assert.equal(out.cells[0].faceOffsets.length, 11);
    // Points are never pruned.
    assert.equal(out.points.length, twoHexes.points.length);
});

step('agglomerate: targetGroupSize=1 is an identity grouping', () => {
    const out = m.agglomerate(twoHexes, 1);
    assert.equal(out.cells[0].cellOffsets.length, 3);  // 2 singleton groups
});

step('agglomerate rejects a non-manifold mesh', () => {
    const tets = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 1, 1, 1]),
        dim: 3,
        cells: [
            {
                type: 'tetra',
                data: new Int32Array([0, 1, 2, 3, 0, 2, 1, 4, 0, 1, 2, 5]),
                nodesPerCell: 4,
            },
        ],
        point_data: {},
        cell_data: {},
        field_data: {},
    };
    assert.throws(() => m.agglomerate(tets));
});

step('agglomerate is reachable as a convertSurfaceOps pipeline step', () => {
    m.writeMesh('/agg.vtu', twoHexes);
    const out = m.convertSurfaceOps('/agg.vtu', '/agg.vtp', [
        { op: 'agglomerate', targetGroupSize: 2 },
    ]);
    assert.equal(out.steps[0].op, 'agglomerate');
    const rendered = m.readMesh('/agg.vtp');
    assert.ok(rendered.cells[0].data.length > 0);
});

step('refine: one hexahedron -> 8 hexahedra with 27 nodes', () => {
    const out = m.refine(cube);
    assert.equal(out.cells.length, 1);
    assert.equal(out.cells[0].type, 'hexahedron');
    assert.equal(out.cells[0].data.length, 8 * 8);
    // 8 corners + 12 edge mids + 6 face centres + 1 body centre.
    assert.equal(out.points.length, 27 * 3);
});

step('refine levels=2 matches refining twice', () => {
    const direct = m.refine(tet, 2);
    const twice = m.refine(m.refine(tet));
    assert.equal(direct.points.length, twice.points.length);
    assert.deepEqual(Array.from(direct.cells[0].data), Array.from(twice.cells[0].data));
});

step('refine rejects a cell type with no same-type subdivision', () => {
    const up = m.convertCells(tet, 'elevate');
    assert.throws(() => m.refine(up));
});

step('refine: a selection is closed up conformingly, not propagated', () => {
    // A 4 x 4 grid of quadrilaterals; refine one cell.
    const n = 4;
    const points = [];
    for (let j = 0; j <= n; ++j)
        for (let i = 0; i <= n; ++i) points.push(i, j, 0);
    const conn = [];
    for (let j = 0; j < n; ++j)
        for (let i = 0; i < n; ++i) {
            const a = j * (n + 1) + i;
            conn.push(a, a + 1, a + n + 2, a + n + 1);
        }
    const grid = {
        points: Float64Array.from(points),
        dim: 3,
        cells: [{ type: 'quad', data: Int32Array.from(conn), nodesPerCell: 4 }],
        point_data: {},
        cell_data: {},
        field_data: {},
    };

    const uniform = m.refine(grid);
    assert.equal(uniform.cells[0].data.length / 4, 64);

    const selective = m.refine(grid, 1, false, { cells: [5], recordLevels: true });
    assert.equal(selective.cells[0].type, 'quad', 'green quads stay quads');
    const nsel = selective.cells[0].data.length / 4;
    assert.ok(nsel > 16 && nsel < 64, `expected a local refinement, got ${nsel} cells`);
    assert.ok('refine:level' in selective.cell_data, 'recordLevels attaches refine:level');

    // Propagation is the always-works baseline: on a connected mesh it reaches
    // every cell, which is exactly the uniform refinement.
    const propagated = m.refine(grid, 1, false, { cells: [5], closure: 'propagate' });
    assert.equal(propagated.cells[0].data.length / 4, 64);

    // Two selectors at once is an error, surfaced as a catchable JS Error.
    assert.throws(() => m.refine(grid, 1, false, { cells: [5], region: 'nope' }));

    // And through the pipeline, which is path-based. The comparison key there is
    // `compare` because `op` is the step's own discriminant.
    m.writeMesh('/grid.vtu', grid);
    const piped = m.convertSurfaceOps('/grid.vtu', '/grid-ref.vtp', [
        { op: 'refine', cells: [5] },
    ]);
    assert.equal(piped.steps.length, 1);
    assert.equal(piped.steps[0].op, 'refine');
    const back = m.readMesh('/grid-ref.vtp');
    assert.ok(back.cells[0].data.length > 0);
});

step('refine: recordHierarchy attaches the persistent parent/child ids', () => {
    const plain = m.refine(cube, 1, false, { recordHierarchy: false });
    assert.ok(!('refine:cell_id' in plain.cell_data), 'not recorded unless asked');

    const hier = m.refine(cube, 1, false, { recordHierarchy: true });
    assert.ok('refine:cell_id' in hier.cell_data);
    assert.ok('refine:parent_id' in hier.cell_data);
    const ids = Array.from(hier.cell_data['refine:cell_id'][0]);
    const parents = Array.from(hier.cell_data['refine:parent_id'][0]);
    assert.equal(new Set(ids).size, ids.length, 'ids are unique');
    // The whole cube is one cell, uniformly refined into 8 -- every child
    // therefore shares parent 0, none can be self-parented (untouched).
    assert.ok(parents.every((p) => p === 0));
    assert.ok(ids.every((id, i) => id !== parents[i]));
    // Also proves the multigrid-stencil fix: redgreen leaves no hanging
    // nodes, so refine:entity would normally never be attached at all.
    assert.ok('refine:entity' in hier.point_data);

    // And through the pipeline (PascalCase keys, generic dispatch).
    m.writeMesh('/cube.vtu', cube);
    const piped = m.convertSurfaceOps('/cube.vtu', '/cube-hier.vtu', [
        { op: 'refine', recordHierarchy: true },
    ]);
    assert.equal(piped.steps.length, 1);
    const backHier = m.readMesh('/cube-hier.vtu');
    assert.ok('refine:cell_id' in backHier.cell_data);
});

step('undoGreen: restores the coarse parent verbatim, read from coarse', () => {
    // A 4 x 4 grid of quadrilaterals; refine one cell selectively so its
    // neighbours pick up transitional (green) closures.
    const n = 4;
    const points = [];
    for (let j = 0; j <= n; ++j)
        for (let i = 0; i <= n; ++i) points.push(i, j, 0);
    const conn = [];
    for (let j = 0; j < n; ++j)
        for (let i = 0; i < n; ++i) {
            const a = j * (n + 1) + i;
            conn.push(a, a + 1, a + n + 2, a + n + 1);
        }
    const coarse = {
        points: Float64Array.from(points),
        dim: 3,
        cells: [{ type: 'quad', data: Int32Array.from(conn), nodesPerCell: 4 }],
        point_data: {},
        cell_data: {},
        field_data: {},
    };

    const fine = m.refine(coarse, 1, false, {
        cells: [5],
        recordHierarchy: true,
        recordLevels: true,
    });
    const fineCells = fine.cells[0].data.length / 4;
    const coarseCells = coarse.cells[0].data.length / 4;

    const undone = m.undoGreen(coarse, fine);
    assert.ok(undone.numGroupsUndone > 0, 'at least one green group was undone');
    assert.ok(undone.numCellsRemoved > 0);
    const undoneCells = undone.mesh.cells[0].data.length / 4;
    assert.ok(undoneCells < fineCells, 'fewer cells than the fine mesh');
    assert.ok(undoneCells > coarseCells, 'more cells than the coarse mesh (red kept)');
    assert.ok(!('refine:cell_id' in undone.mesh.cell_data), 'reserved arrays are dropped');
    assert.ok(!('refine:entity' in undone.mesh.point_data), 'reserved arrays are dropped');

    // Fails by name rather than guessing: no hierarchy at all here.
    assert.throws(() => m.undoGreen(coarse, coarse));

    // It is a two-mesh op, so it is deliberately NOT reachable as a pipeline
    // step -- the same exclusion Merge/Interpolate/Split/Diff already have
    // (it never reaches pipeline_op_table(), so this is the same generic
    // "unknown operation" message those get here, not the C++ engine's more
    // specific excluded-hint text).
    m.writeMesh('/ug-coarse.vtu', coarse);
    assert.throws(
        () => m.convertSurfaceOps('/ug-coarse.vtu', '/ug-out.vtu', [{ op: 'undoGreen' }]),
        /unknown operation 'undoGreen'/,
    );
});

step('decimate: collapses a refined cube skin, pinning its creases', () => {
    // The skin of a refined cube: 24 quads -> 48 triangles, with every cube
    // edge/corner vertex a pinned feature; only face-interior vertices go.
    const skin = m.extractSkin(m.refine(cube), true);
    const out = m.decimate(skin, 0.5);
    assert.equal(out.mesh.cells.length, 1);
    assert.equal(out.mesh.cells[0].type, 'triangle');
    assert.ok(out.mesh.cells[0].data.length / 3 < 48);
    assert.ok(out.facesRemoved > 0);
    assert.ok(out.pointsRemoved > 0);
    assert.ok(out.collapsesRejected >= 0);
    assert.ok(out.maxErrorApplied >= 0);
});

step('decimate rejects a volume mesh and a missing criterion', () => {
    assert.throws(() => m.decimate(cube, 0.5), /extract_surface/);
    const skin = m.extractSkin(cube, true);
    assert.throws(() => m.decimate(skin));
});

step('decimate is reachable as a convertSurfaceOps pipeline step', () => {
    // The pipeline runs against the loaded mesh itself, so hand it a surface
    // mesh (decimate refuses volume input by design).
    m.writeMesh('/dec.vtu', m.extractSkin(m.refine(cube), true));
    const out = m.convertSurfaceOps('/dec.vtu', '/dec.vtp', [{ op: 'decimate' }]);
    assert.equal(out.steps[0].op, 'decimate');
    assert.equal(typeof out.steps[0].facesRemoved, 'number');
    assert.ok(m.readMesh('/dec.vtp').cells[0].data.length / 3 < 48);
});

step('remesh: produces the requested cluster count on a closed surface', () => {
    // Unlike every other geometry op, the output has NO correspondence to
    // the input -- new points, new connectivity -- so it is asserted purely
    // on shape/counters, never against the input's own point/cell ids.
    const skin = m.extractSkin(cube, true);
    const out = m.remesh(skin, 40);
    assert.equal(out.mesh.cells.length, 1);
    assert.equal(out.mesh.cells[0].type, 'triangle');
    assert.equal(out.numClusters, 40);
    assert.equal(out.mesh.points.length / 3, 40);
    assert.ok(out.subdivideApplied > 0, 'the cube skin cannot support 40 clusters unsubdivided');
    assert.ok(out.numIterations >= 0);
    assert.equal(out.numIsolatedClusters, 0);
    assert.ok(out.numNonManifoldVertices >= 0);

    // The quadric ("feature-preserving") metric is accepted too.
    const outQ = m.remesh(skin, 40, -1, 10.0, 4, 100, 10, 'quadric');
    assert.equal(outQ.numClusters, 40);

    // gradation/preserveBoundary reach the C++ core and are honoured.
    const outG = m.remesh(skin, 40, -1, 10.0, 4, 100, 10, 'isotropic', 1.5, false);
    assert.equal(outG.numClusters, 40);

    // The anisotropic metric + maxAnisotropy are accepted too.
    const outA = m.remesh(skin, 40, -1, 10.0, 4, 100, 10, 'anisotropic', 0.0, true, 3.0);
    assert.equal(outA.mesh.cells[0].type, 'triangle');
    assert.ok(outA.numClusters > 0);
});

step('remesh rejects a volume mesh and a too-small cluster count', () => {
    assert.throws(() => m.remesh(cube, 10), /extract_surface/);
    const skin = m.extractSkin(cube, true);
    assert.throws(() => m.remesh(skin, 3));
});

step('remesh is reachable as a convertSurfaceOps pipeline step', () => {
    m.writeMesh('/remesh.vtu', m.extractSkin(cube, true));
    const out = m.convertSurfaceOps('/remesh.vtu', '/remesh.vtp', [
        { op: 'remesh', numClusters: 30, metric: 'quadric', gradation: 1.0 },
    ]);
    assert.equal(out.steps[0].op, 'remesh');
    assert.equal(out.steps[0].numClusters, 30);
    assert.equal(m.readMesh('/remesh.vtp').points.length / 3, 30);

    // MaxAnisotropy joins Gradation/PreserveBoundary in the same step key
    // list -- reachable through the identical generic dispatch.
    const outA = m.convertSurfaceOps('/remesh.vtu', '/remesh_aniso.vtp', [
        { op: 'remesh', numClusters: 30, metric: 'anisotropic', maxAnisotropy: 3.0 },
    ]);
    assert.equal(outA.steps[0].op, 'remesh');
    assert.ok(m.readMesh('/remesh_aniso.vtp').points.length / 3 > 0);
});

step('remeshVolume: retetrahedralizes a volume mesh directly', () => {
    // Unlike remesh, remeshVolume accepts a VOLUME mesh directly (no
    // extractSkin needed first) -- its boundary is extracted internally.
    const out = m.remeshVolume(cube, null, 0.4, null, 0, 0.1, 20000000, 20000000, 0.35,
        'pseudonormal', 'off');
    assert.equal(out.mesh.cells.length, 1);
    assert.equal(out.mesh.cells[0].type, 'tetra');
    assert.ok(out.numTets > 0);
    assert.equal(out.numTets, out.mesh.cells[0].data.length / 4);
    assert.ok(out.numVerticesWarped >= 0);
    assert.ok(out.numTetsRejected >= 0);
    assert.ok(out.numNonManifoldEdges >= 0);

    // Also accepts a closed SURFACE directly, extractSkin's own output.
    const skin = m.extractSkin(cube, true);
    const outFromSurface = m.remeshVolume(skin, null, 0.4, null, 0, 0.1, 20000000, 20000000, 0.35,
        'pseudonormal', 'off');
    assert.ok(outFromSurface.numTets > 0);
});

step('remeshVolume is reachable as a convertSurfaceOps pipeline step', () => {
    // convertSurfaceOps always ends on a boundary extraction (it is the
    // viewer's rendering pipeline), so the VTP output is the tet mesh's own
    // triangle boundary, not the tet mesh itself -- unlike the plain
    // runPipeline test above, which writes exactly what the pipeline
    // produced. The step's own counters are what actually prove the op ran.
    m.writeMesh('/rvol.vtu', cube);
    const out = m.convertSurfaceOps('/rvol.vtu', '/rvol.vtp', [
        { op: 'remeshVolume', cellSize: 0.4, watertightCheck: 'off' },
    ]);
    assert.equal(out.steps[0].op, 'remeshVolume');
    assert.ok(out.steps[0].numTets > 0);
    const rendered = m.readMesh('/rvol.vtp');
    assert.equal(rendered.cells[0].type, 'triangle');
    assert.ok(rendered.cells[0].data.length > 0);
});

step('computeCurvature satisfies Gauss-Bonnet on a closed surface', () => {
    // A cube's boundary skin: closed, consistently wound, so the angle
    // defects sum to 2*pi*chi = 4*pi exactly, whatever the tessellation.
    const skin = m.extractSurface(cube);
    const out = m.computeCurvature(skin, true, true, 'mixed-voronoi', false, true, true, '');
    assert.ok(Math.abs(out.totalAngleDefect - 4 * Math.PI) < 1e-9);
    assert.equal(out.numBoundary, 0);
    assert.equal(out.numIsolated, 0);
    assert.equal(out.numDegenerate, 0);
    assert.equal(out.quality.watertight, true);
    assert.equal(out.quality.inconsistentPairs, 0);
    assert.equal(out.mesh.points.length, skin.points.length); // a pure data step
    // Barycentric gives the same defect: it changes the dual area, and the
    // defect never touches it.
    const bary = m.computeCurvature(skin, true, true, 'barycentric');
    assert.equal(bary.totalAngleDefect, out.totalAngleDefect);
    // An unknown dual area is refused by name, not silently defaulted.
    assert.throws(() => m.computeCurvature(skin, true, true, 'nope'));
});

step('computeCurvature is reachable as a convertSurfaceOps pipeline step', () => {
    const skin = m.extractSurface(cube);
    m.writeMesh('/curv.vtu', skin);
    const out = m.convertSurfaceOps('/curv.vtu', '/curv.vtp', [
        { op: 'curvature', recordPrincipal: true },
    ]);
    assert.equal(out.steps[0].op, 'curvature');
    assert.ok(Math.abs(out.steps[0].totalAngleDefect - 4 * Math.PI) < 1e-9);
});

step('optimizeVolume: ODT-remeshes a tetrahedral mesh', () => {
    // A tetra mesh (simplexified hex cube). optimizeVolume relocates vertices
    // and flips connectivity; the boundary is preserved and no cell inverts.
    const tetCube = m.convertCells(cube, 'simplexify');
    const out = m.optimizeVolume(tetCube, 5, true, true, true, 1e-6);
    assert.equal(out.mesh.cells.length, 1);
    assert.equal(out.mesh.cells[0].type, 'tetra');
    assert.ok(out.numTets > 0);
    assert.equal(out.mesh.points.length, tetCube.points.length); // point set invariant
    assert.ok(out.minQualityAfter >= out.minQualityBefore - 1e-9); // monotone
    assert.ok(out.numFlips >= 0 && out.num23Flips >= 0 && out.num32Flips >= 0);
});

step('optimizeVolume is reachable as a convertSurfaceOps pipeline step', () => {
    const tetCube = m.convertCells(cube, 'simplexify');
    m.writeMesh('/ovol.vtu', tetCube);
    const out = m.convertSurfaceOps('/ovol.vtu', '/ovol.vtp', [
        { op: 'optimizeVolume', maxIterations: 3 },
    ]);
    assert.equal(out.steps[0].op, 'optimizeVolume');
    assert.ok(out.steps[0].numTets > 0);
});

step('subdivide is reachable as a convertSurfaceOps pipeline step', () => {
    m.writeMesh('/sub.vtu', cube);
    const out = m.convertSurfaceOps('/sub.vtu', '/sub.vtp', [{ op: 'subdivide' }]);
    assert.equal(out.steps[0].op, 'subdivide');
    // Every internal face subdivide adds is shared by exactly two children
    // and cancels out of the boundary, so the rendered surface is geometrically
    // the same box as the input -- this asserts the pipeline step actually ran
    // (a genuinely different, non-empty cell block came back), not a specific
    // facet count.
    const rendered = m.readMesh('/sub.vtp');
    assert.ok(rendered.cells[0].data.length > 0);
});

step('smooth: relaxes an interior node while pinning the boundary', () => {
    // A refined cube is a 3x3x3 node lattice: exactly one node (the body
    // centre) is interior, so it is the only one smoothing may move.
    const grid = m.refine(cube);
    const points = Float64Array.from(grid.points);
    const inside = (v) => v > 1e-9 && v < 1 - 1e-9;
    let interior = -1;
    for (let i = 0; i < points.length / 3; ++i) {
        if (inside(points[3 * i]) && inside(points[3 * i + 1]) && inside(points[3 * i + 2])) {
            interior = i;
            break;
        }
    }
    assert.notEqual(interior, -1, 'the refined cube should have an interior node');

    // Pull it off its own centroid so there is something to relax.
    points[3 * interior] += 0.2;
    const perturbed = { ...grid, points };

    const out = m.smooth(perturbed, 'laplacian', 5);
    // Geometry only: same point count, same connectivity, moved coordinates.
    assert.equal(out.mesh.points.length, perturbed.points.length);
    assert.equal(out.mesh.cells.length, perturbed.cells.length);
    assert.deepEqual(
        Array.from(out.mesh.cells[0].data),
        Array.from(perturbed.cells[0].data),
    );
    assert.notEqual(out.mesh.points[3 * interior], perturbed.points[3 * interior]);
    // The interior node is pulled back toward the centre it was moved from.
    assert.ok(
        Math.abs(out.mesh.points[3 * interior] - 0.5) <
            Math.abs(perturbed.points[3 * interior] - 0.5),
    );
    // Every boundary node is pinned, so only that one node moved.
    for (let i = 0; i < points.length / 3; ++i) {
        if (i === interior) continue;
        for (let c = 0; c < 3; ++c)
            assert.equal(out.mesh.points[3 * i + c], perturbed.points[3 * i + c]);
    }

    assert.equal(out.numNodesMoved, 1);
    assert.ok(out.maxDisplacement > 0);
    assert.equal(typeof out.numSkippedInversion, 'number');
});

step('smooth: taubin defaults leave a structured hex block alone', () => {
    // The negative default lambda means "the method's own default" and must
    // reach the core unchanged; an already-relaxed lattice is a fixed point.
    const grid = m.refine(cube);
    const out = m.smooth(grid);
    assert.equal(out.numNodesMoved, 0);
    assert.deepEqual(Array.from(out.mesh.points), Array.from(grid.points));
});

step('smooth rejects an unknown method', () => {
    assert.throws(() => m.smooth(cube, 'not-a-method'));
});

step("smooth: method 'odt' moves a tet mesh's free vertices", () => {
    // ODT is tet-only, unlike taubin/laplacian, which run on the hex cube.
    const tets = m.convertCells(cube, 'simplexify');
    const out = m.smooth(tets, 'odt', 3, -1, -0.34, false, false, 30, true);
    assert.equal(out.mesh.cells[0].type, 'tetra');
    assert.equal(out.mesh.points.length, tets.points.length);
    assert.ok(out.numNodesMoved >= 0);

    // ODT refuses a non-tet block by name, rather than silently ignoring it.
    assert.throws(() => m.smooth(cube, 'odt'), /tet/);
});

step('interpolate: transfers fields onto a target mesh', () => {
    // Nearest: the single target point sits next to the second source point.
    const src = {
        points: new Float64Array([0, 0, 0, 1, 0, 0]),
        dim: 3,
        cells: [],
        point_data: { f: new Float64Array([10, 20]) },
        cell_data: {},
        field_data: {},
    };
    const tgt = {
        points: new Float64Array([0.9, 0, 0]),
        dim: 3,
        cells: [],
        point_data: {},
        cell_data: {},
        field_data: {},
    };
    const out = m.interpolate(src, tgt);
    assert.deepEqual(Array.from(out.point_data.f), [20]);

    // Barycentric is exact on a linear field (g = x at the cube's corners).
    const cubeSrc = { ...cube, point_data: { g: new Float64Array([0, 1, 1, 0, 0, 1, 1, 0]) } };
    const probe = {
        points: new Float64Array([0.25, 0.5, 0.5]),
        dim: 3,
        cells: [],
        point_data: {},
        cell_data: {},
        field_data: {},
    };
    const bar = m.interpolate(cubeSrc, probe, 'barycentric');
    assert.ok(Math.abs(bar.point_data.g[0] - 0.25) < 1e-12);

    assert.throws(() => m.interpolate(src, tgt, 'not-a-method'));
    assert.throws(() => m.interpolate(src, tgt, 'nearest', ['nope']));
});

step('conservativeInterpolate: mass-preservingly transfers cell_data', () => {
    // Self-remap of cell_data onto the identical triangle exactly recovers
    // the source value (100% coverage, no clipping loss).
    const tri = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        dim: 3,
        cells: [{ type: 'triangle', data: new Int32Array([0, 1, 2]), nodesPerCell: 3 }],
        point_data: {},
        cell_data: { f: [new Float64Array([7])] },
        field_data: {},
    };
    const out = m.conservativeInterpolate(tri, tri, ['f'], 0, 'suffix');
    assert.deepEqual(Array.from(out.cell_data.f_interp[0]), [7]);

    assert.throws(() => m.conservativeInterpolate(tri, tri, ['nope']));
});

step('slice: the mid-plane cross-section of the unit cube has area 1', () => {
    const sectioned = m.slice(cube, [0, 0, 0.5], [0, 0, 1], true);
    // A volume mesh sections into surface faces (triangles here).
    let area = 0;
    let nfaces = 0;
    for (const block of sectioned.cells) {
        const npc = block.nodesPerCell;
        const data = block.data;
        for (let c = 0; c < data.length / npc; ++c) {
            nfaces += 1;
            const p = [];
            for (let k = 0; k < npc; ++k) {
                const n = data[c * npc + k];
                p.push([sectioned.points[n * 3], sectioned.points[n * 3 + 1]]);
            }
            // Shoelace on the projected (x, y) ring — the section lies at z = 0.5.
            let s = 0;
            for (let k = 0; k < p.length; ++k) {
                const a = p[k];
                const b = p[(k + 1) % p.length];
                s += a[0] * b[1] - b[0] * a[1];
            }
            area += Math.abs(s) / 2;
        }
    }
    assert.ok(nfaces > 0, 'section is non-empty');
    assert.ok(Math.abs(area - 1) < 1e-9, `section area ${area} != 1`);
    assert.ok('slice:parent_cell' in sectioned.cell_data, 'parent ids attached');
    // A plane missing the mesh yields an empty section.
    const empty = m.slice(cube, [0, 0, 5], [0, 0, 1], false);
    assert.equal(empty.cells.length, 0);
    assert.throws(() => m.slice(cube, [0, 0, 0], [0, 0, 0]));
});

step('isosurface: the f = z level set of the unit cube has area 1', () => {
    // f = z on the cube's 8 corners, so the isosurface at z = 0.5 is the same
    // unit square slice() cuts there -- the data-driven route to it.
    const field = { ...cube, point_data: { f: new Float64Array([0, 0, 0, 0, 1, 1, 1, 1]) } };
    const iso = m.isosurface(field, 'f', 0.5, -1, true);
    let area = 0;
    let nfaces = 0;
    for (const block of iso.cells) {
        const npc = block.nodesPerCell;
        const data = block.data;
        for (let c = 0; c < data.length / npc; ++c) {
            nfaces += 1;
            let s = 0;
            for (let k = 0; k < npc; ++k) {
                const a = data[c * npc + k];
                const b = data[c * npc + ((k + 1) % npc)];
                s += iso.points[a * 3] * iso.points[b * 3 + 1] -
                     iso.points[b * 3] * iso.points[a * 3 + 1];
            }
            area += Math.abs(s) / 2;
        }
    }
    assert.ok(nfaces > 0, 'contour is non-empty');
    assert.ok(Math.abs(area - 1) < 1e-9, `contour area ${area} != 1`);
    // The contoured field reads back as exactly the isovalue.
    for (const v of iso.point_data.f) assert.equal(v, 0.5);
    assert.ok('iso:parent_cell' in iso.cell_data, 'parent ids attached');
    assert.ok('iso:value' in iso.cell_data && 'iso:index' in iso.cell_data, 'contours tagged');

    // Two isovalues land in one mesh, tagged and in ascending order.
    const two = m.isosurface(field, 'f', [0.75, 0.25]);
    const vals = [].concat(...two.cell_data['iso:value'].map((b) => Array.from(b)));
    assert.deepEqual([...new Set(vals)], [0.25, 0.75], 'ascending, tagged');

    // Out of range is an empty contour, not an error; a cell field has no level set.
    assert.equal(m.isosurface(field, 'f', 5).cells.length, 0);
    assert.throws(() => m.isosurface(cube, 'nope', 0.5));
});

step('gradient: a linear field is differentiated exactly, and the (n,3) shape survives', () => {
    // A frustum, not the unit cube: on a cube every face is a parallelogram
    // whose corner average IS its area centroid, so the exactness assertion
    // below would pass even with a broken quadrature.
    const frustum = {
        points: new Float64Array([
            0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0,
            0.5, 0.5, 1, 1.5, 0.5, 1, 1.5, 1.5, 1, 0.5, 1.5, 1,
        ]),
        dim: 3,
        cells: [
            { type: 'hexahedron', data: new Int32Array([0, 1, 2, 3, 4, 5, 6, 7]), nodesPerCell: 8 },
        ],
        point_data: {},
        cell_data: {},
        field_data: {},
    };
    const f = new Float64Array(8);
    for (let i = 0; i < 8; ++i) {
        f[i] = 3 * frustum.points[i * 3] - 2 * frustum.points[i * 3 + 1] +
               5 * frustum.points[i * 3 + 2] + 7;
    }
    const field = { ...frustum, point_data: { f } };

    const g = m.gradient(field, 'f');
    assert.equal(g.numSkipped, 0);
    assert.equal(g.numFallback, 0);
    const grad = g.mesh.cell_data['f:gradient'][0];
    assert.equal(grad.length, 3, 'one cell x 3 components');
    assert.ok(Math.abs(grad[0] - 3) < 1e-12, `d/dx ${grad[0]} != 3`);
    assert.ok(Math.abs(grad[1] + 2) < 1e-12, `d/dy ${grad[1]} != -2`);
    assert.ok(Math.abs(grad[2] - 5) < 1e-12, `d/dz ${grad[2]} != 5`);

    // THE COMPONENT-COUNT ROUND-TRIP (v9.9.0's *_components maps). A flat typed
    // array carries no shape, so without the sibling map an (n, 3) gradient
    // re-enters C++ as (3n, 1). This must cross the OBJECT boundary -- a
    // path-based call never materializes a JS mesh and so cannot detect it.
    assert.equal(g.mesh.cell_data_components['f:gradient'], 3,
                 'the gradient declares 3 components');
    const back = m.gradient(g.mesh, 'f', 'gradient', 'green-gauss', 'cell', 'again');
    assert.equal(back.mesh.cell_data_components['f:gradient'], 3,
                 'the component count survives a round trip through JS');
    assert.equal(back.mesh.cell_data['f:gradient'][0].length, 3,
                 'the array is still one row of 3, not three rows of 1');

    // A 3-component input yields the (n, 9) tensor, declared as 9 components.
    const u = new Float64Array(24);
    for (let i = 0; i < 8; ++i) {
        const x = frustum.points[i * 3], y = frustum.points[i * 3 + 1], z = frustum.points[i * 3 + 2];
        u[i * 3] = 7 * z;
        u[i * 3 + 1] = 11 * x;
        u[i * 3 + 2] = 13 * y;
    }
    const vec = { ...frustum, point_data: { u }, point_data_components: { u: 3 } };
    const tensor = m.gradient(vec, 'u');
    assert.equal(tensor.mesh.cell_data_components['u:gradient'], 9, '(n, 9) tensor');
    assert.equal(tensor.mesh.cell_data['u:gradient'][0].length, 9);

    // curl of (7z, 11x, 13y) is (13, 7, 11): three distinct nonzero components,
    // so any index permutation or sign flip fails.
    const curl = m.gradient(vec, 'u', 'curl');
    const c = curl.mesh.cell_data['u:curl'][0];
    assert.equal(curl.mesh.cell_data_components['u:curl'], 3);
    assert.ok(Math.abs(c[0] - 13) < 1e-12 && Math.abs(c[1] - 7) < 1e-12 &&
              Math.abs(c[2] - 11) < 1e-12, `curl ${Array.from(c)} != 13,7,11`);

    // The point location moves the result into point_data, declared likewise.
    const atPoints = m.gradient(field, 'f', 'gradient', 'green-gauss', 'point');
    assert.equal(atPoints.mesh.point_data_components['f:gradient'], 3);
    assert.equal(atPoints.mesh.point_data['f:gradient'].length, 8 * 3);

    // Least squares on a lone cell has no neighbourhood, so the fallback fires.
    const lsq = m.gradient(field, 'f', 'gradient', 'least-squares');
    assert.equal(lsq.numFallback, 1, 'a cell with no neighbours must fall back');

    // A cell_data field has no derivative, and a scalar has no divergence.
    assert.throws(() => m.gradient(g.mesh, 'f:gradient'));
    assert.throws(() => m.gradient(field, 'f', 'divergence'));
});

step('hessian: exactly zero for a linear field, (n,9), and scalar-only', () => {
    // The exact same frustum fixture the gradient step above builds:
    // f = 3x - 2y + 5z + 7 (linear, so its Hessian is exactly zero
    // everywhere -- the one mesh-shape-independent guarantee), u a genuine
    // 3-vector (rejected: hessian is scalar-only).
    const frustum = {
        points: new Float64Array([
            0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0,
            0.5, 0.5, 1, 1.5, 0.5, 1, 1.5, 1.5, 1, 0.5, 1.5, 1,
        ]),
        dim: 3,
        cells: [
            { type: 'hexahedron', data: new Int32Array([0, 1, 2, 3, 4, 5, 6, 7]), nodesPerCell: 8 },
        ],
        point_data: {},
        cell_data: {},
        field_data: {},
    };
    const f = new Float64Array(8);
    for (let i = 0; i < 8; ++i) {
        f[i] = 3 * frustum.points[i * 3] - 2 * frustum.points[i * 3 + 1] +
               5 * frustum.points[i * 3 + 2] + 7;
    }
    const field = { ...frustum, point_data: { f } };
    const u = new Float64Array(24);
    const vec = { ...frustum, point_data: { u }, point_data_components: { u: 3 } };

    const h = m.hessian(field, 'f');
    assert.equal(h.numSkipped, 0);
    const hess = h.mesh.cell_data['f:hessian'][0];
    assert.equal(hess.length, 9, 'one cell x 9 components');
    assert.equal(h.mesh.cell_data_components['f:hessian'], 9,
                 'the hessian declares 9 components');
    for (let i = 0; i < 9; ++i)
        assert.ok(Math.abs(hess[i]) < 1e-9, `hessian[${i}] = ${hess[i]} != 0`);

    // The point location moves the result into point_data, declared likewise.
    const atPoints = m.hessian(field, 'f', 'green-gauss', 'point');
    assert.equal(atPoints.mesh.point_data_components['f:hessian'], 9);
    assert.equal(atPoints.mesh.point_data['f:hessian'].length, 8 * 9);

    // A cell_data field has no derivative, and a vector field is scalar-only.
    assert.throws(() => m.hessian(h.mesh, 'f:hessian'));
    assert.throws(() => m.hessian(vec, 'u'));
});

step('estimateError: zero on a linear field, nonzero and markable on a quadratic one', () => {
    // Two unit-cube hexahedra stacked along z, sharing the z=1 face -- a
    // single-cell mesh cannot show a nonzero indicator at all, since
    // averaging one cell's own value back onto itself is a no-op.
    const pts = new Float64Array([
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
        0, 0, 2, 1, 0, 2, 1, 1, 2, 0, 1, 2,
    ]);
    const conn = new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11]);
    const twoCells = {
        points: pts, dim: 3,
        cells: [{ type: 'hexahedron', data: conn, nodesPerCell: 8 }],
        point_data: {}, cell_data: {}, field_data: {},
    };

    // Linear field: raw == recovered everywhere, so the indicator is zero.
    const lin = new Float64Array(12);
    for (let i = 0; i < 12; ++i) lin[i] = 2 * pts[i * 3] - 3 * pts[i * 3 + 1] + 0.5 * pts[i * 3 + 2];
    const linField = { ...twoCells, point_data: { f: lin } };
    const e0 = m.estimateError(linField, 'f');
    assert.equal(e0.numSkipped, 0);
    assert.ok(e0.globalError < 1e-9, `global_error ${e0.globalError} should be ~0 on a linear field`);
    assert.ok(e0.mesh.cell_data['error:zz'][0].every((v) => Math.abs(v) < 1e-9));
    assert.equal(e0.mesh.cell_data['error:marked'], undefined, 'no marking requested');

    // Quadratic field: the two cells' raw gradients genuinely differ, so
    // recovery (averaging across the shared face) gives a nonzero indicator.
    const quad = new Float64Array(12);
    for (let i = 0; i < 12; ++i) {
        const x = pts[i * 3], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
        quad[i] = x * x + y * y + z * z;
    }
    const quadField = { ...twoCells, point_data: { f: quad } };
    const e1 = m.estimateError(quadField, 'f', 'zz', 'absolute', 1e-9, '', '', true);
    assert.equal(e1.numSkipped, 0);
    assert.ok(e1.globalError > 0, 'a quadratic field must give a nonzero global error');
    assert.equal(e1.mesh.cell_data['error:marked'].length, 1, 'one marked block per cell block');
    assert.ok(e1.mesh.cell_data['error:marked'][0].every((v) => v === 0 || v === 1));

    // A cell_data field has no derivative to recover; an out-of-range
    // marking_value for "fraction" is rejected.
    assert.throws(() => m.estimateError(e1.mesh, 'error:zz'));
    assert.throws(() => m.estimateError(quadField, 'f', 'zz', 'fraction', 1.5));
});

step('convertSurfaceOps: gradient is a chainable pipeline step', () => {
    // The pipeline is path-based and re-skins at the end, so a POINT-located
    // gradient is the one that survives to the written surface.
    const f = new Float64Array([0, 0, 0, 0, 1, 1, 1, 1]);
    m.writeMesh('/grad.vtu', { ...cube, point_data: { f } });
    const rep = m.convertSurfaceOps('/grad.vtu', '/grad.vtp', [
        { op: 'gradient', array: 'f', location: 'point', output: 'gf' },
    ]);
    assert.equal(rep.steps[0].op, 'gradient');
    assert.equal(rep.steps[0].numSkipped, 0);
    const skin = m.readMesh('/grad.vtp');
    assert.ok('gf' in skin.point_data, 'the gradient rides through the re-skin');
    // f = z on the cube, so the gradient is (0, 0, 1) everywhere.
    assert.equal(skin.point_data_components.gf, 3);
    for (let i = 0; i < skin.point_data.gf.length / 3; ++i) {
        assert.ok(Math.abs(skin.point_data.gf[i * 3 + 2] - 1) < 1e-9,
                  `d/dz ${skin.point_data.gf[i * 3 + 2]} != 1`);
    }
});

step('convertSurfaceOps: hessian is a chainable pipeline step', () => {
    // f = z on the cube is linear, so its Hessian is exactly zero -- the one
    // mesh-shape-independent guarantee, reused as the pipeline oracle too.
    const f = new Float64Array([0, 0, 0, 0, 1, 1, 1, 1]);
    m.writeMesh('/hess.vtu', { ...cube, point_data: { f } });
    const rep = m.convertSurfaceOps('/hess.vtu', '/hess.vtp', [
        { op: 'hessian', array: 'f', location: 'point', output: 'hf' },
    ]);
    assert.equal(rep.steps[0].op, 'hessian');
    assert.equal(rep.steps[0].numSkipped, 0);
    const skin = m.readMesh('/hess.vtp');
    assert.ok('hf' in skin.point_data, 'the hessian rides through the re-skin');
    assert.equal(skin.point_data_components.hf, 9);
    for (let i = 0; i < skin.point_data.hf.length; ++i)
        assert.ok(Math.abs(skin.point_data.hf[i]) < 1e-9, `hf[${i}] = ${skin.point_data.hf[i]} != 0`);
});

step('partition: refined hexahedra decompose into 2 balanced pieces', () => {
    const grid = m.refine(cube);  // 8 hexahedra
    const pieces = m.partition(grid, 2);
    assert.equal(pieces.length, 2);
    assert.equal(pieces[0].partId, 0);
    assert.equal(pieces[1].partId, 1);
    let total = 0;
    for (const piece of pieces) {
        // Blocks are kept 1:1 with the input (unlike split).
        assert.equal(piece.mesh.cells.length, grid.cells.length);
        total += piece.mesh.cells[0].data.length / 8;
    }
    assert.equal(total, 8);

    const labels = m.partitionLabels(grid, 2);
    assert.equal(labels.length, 1);
    assert.equal(labels[0].length, 8);
    for (const p of labels[0]) {
        assert.ok(p >= 0 && p < 2);
    }
});

step('partition: kahip is compiled out of the WASM build and says so', () => {
    assert.throws(() => m.partition(cube, 2, 'kahip'), /MESHIOPLUSPLUS_WITH_KAHIP/);
});

// The unit cube as a closed, outward-wound triangle surface: the distance
// bindings need a watertight surface, and the other fixtures here are volumes.
const cubeSurface = {
    points: new Float64Array([
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
    ]),
    dim: 3,
    cells: [{
        type: 'triangle',
        data: new Int32Array([
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
        ]),
        nodesPerCell: 3,
    }],
    point_data: {},
    cell_data: {},
    field_data: {},
};

step('grid builds a lattice from nothing', () => {
    const g = m.grid([2, 2, 2]);
    assert.equal(g.points.length, 27 * 3);
    assert.equal(g.cells.length, 1);
    assert.equal(g.cells[0].type, 'hexahedron');
    // An empty lattice is a legal request, not a throw.
    assert.equal(m.grid([0, 0, 0]).points.length, 0);
});

step('voxelize keeps the whole box, the shell, or the interior', () => {
    const q = m.surfaceWatertightCheck(cubeSurface);
    assert.equal(q.watertight, true);
    assert.equal(q.boundaryEdges, 0);

    const all = m.voxelize(cubeSurface, [4, 4, 4]);
    assert.equal(all.numOccupied, 64);
    assert.deepEqual(Array.from(all.dims), [4, 4, 4]);
    assert.ok(Math.abs(all.spacing[0] - 0.25) < 1e-12);

    const inside = m.voxelize(cubeSurface, [5, 5, 5], 0,
        [-0.5, -0.5, -0.5, 1.5, 1.5, 1.5], 0, 0, 'inside', 'pseudonormal', false,
        20000000, 'off');
    assert.equal(inside.numOccupied, 27);

    // An unknown fill fails by name rather than silently defaulting.
    assert.throws(() => m.voxelize(cubeSurface, [2, 2, 2], 0, null, 0, 0, 'solid'));
});

step('sampleDistance matches the cube\'s closed form', () => {
    // The centre is 0.5 in; the other two are 1.0 out.
    const d = m.sampleDistance(cubeSurface,
        [0.5, 0.5, 0.5, 2.0, 0.5, 0.5, -1.0, 0.5, 0.5], 'pseudonormal', 0, 'off');
    assert.equal(d.length, 3);
    assert.ok(Math.abs(d[0] + 0.5) < 1e-12, `expected -0.5, got ${d[0]}`);
    assert.ok(Math.abs(d[1] - 1.0) < 1e-12);
    assert.ok(Math.abs(d[2] - 1.0) < 1e-12);
});

step('distanceToSurface attaches sdf:distance as ordinary point data', () => {
    const q = m.grid([2, 2, 2], [-0.5, -0.5, -0.5], [1, 1, 1]);
    const out = m.distanceToSurface(q, cubeSurface, 'pseudonormal', 'corner', 0, true, 'off');
    assert.equal(out.numBanded, 0);
    assert.equal(out.quality.watertight, true);
    assert.ok('sdf:distance' in out.mesh.point_data);
    assert.ok('sdf:inside' in out.mesh.point_data);
});

step('computeSdf generates the grid and the field in one call', () => {
    const g = m.computeSdf(cubeSurface, 'voxel', [4, 4, 4], 0, null, 0, 0.1, 8, 4, 1,
        true, 20000000, 'pseudonormal', 'corner', 0, 'off');
    assert.deepEqual(Array.from(g.dims), [4, 4, 4]);
    assert.equal(g.maxDepth, 0);
    assert.ok('sdf:distance' in g.mesh.point_data);
    assert.equal(g.mesh.cells[0].data.length, 64 * 8);
    // The header rides across as ordinary numeric field_data, so every binding
    // carries it -- but no FORMAT does, which is why .vti exists.
    assert.ok('sdf:origin' in g.mesh.field_data);
    assert.ok('sdf:spacing' in g.mesh.field_data);

    // The octree refines only near the surface.
    const tree = m.computeSdf(cubeSurface, 'octree', null, 0, null, 0, 0.1, 4, 2, 1,
        true, 20000000, 'pseudonormal', 'corner', 0, 'off');
    assert.equal(tree.maxDepth, 2);
    const n = tree.mesh.cells[0].data.length / 8;
    assert.ok(n > 64 && n < 4096, `octree produced ${n} cells`);

    // resolution/cellSize size a voxel grid; an octree's finest cell is already
    // determined, so passing one is an error rather than a preference.
    assert.throws(() => m.computeSdf(cubeSurface, 'octree', [4, 4, 4]));
    assert.throws(() => m.computeSdf(cubeSurface, 'quadtree', [4, 4, 4]));
});

step('cropPredicate keeps the cells a data comparison selects', () => {
    const dom = m.grid([4, 4, 4], [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
    const field = m.distanceToSurface(dom, cubeSurface, 'pseudonormal', 'center', 0, false,
        'off');
    const kept = m.cropPredicate(field.mesh, 'sdf:distance', '<', 0);
    assert.equal(kept.cells[0].data.length, 8 * 8);
    // There is no `mode`: a cell_data value is already one per cell.
    assert.throws(() => m.cropPredicate(field.mesh, 'sdf:distance', '~', 0));
    assert.throws(() => m.cropPredicate(field.mesh, 'nope', '<', 0));
});

step('Voxelize is a chainable pipeline step', () => {
    // It is the one step that REPLACES geometry rather than transforming it:
    // a triangle skin goes in and a hexahedron lattice comes out.
    m.writeMesh('/vox-in.vtu', cubeSurface);
    m.convertSurfaceOps('/vox-in.vtu', '/vox-out.vtu',
        [{ op: 'voxelize', resolution: [3, 3, 3] }]);
    const out = m.readMesh('/vox-out.vtu');
    // convertSurfaceOps skins whatever the chain produces, so a 3x3x3 lattice
    // comes back as its boundary: 6 faces x 9 quads. That the result is quads
    // rather than triangles is itself the evidence the step ran -- the input
    // was a triangle skin.
    assert.equal(out.cells[0].type, 'quad');
    assert.equal(out.cells[0].data.length, 54 * 4);
});

step('provenance is opt-in, scoped, and readable back', () => {
    const mesh = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        dim: 3,
        cells: [{ type: 'triangle', data: new Int32Array([0, 1, 2]), nodesPerCell: 3 }],
    };

    // Default (no scope): exactly the one-line credit, unchanged since v10.15.0.
    m.writeMesh('/off.obj', mesh, 'obj');
    const off = m.readProvenance('/off.obj');
    assert(off.recognised, 'default write should still carry the credit line');
    assert(off.lines.length === 1, `default write leaked a block: ${off.lines}`);

    // Scoped: the full block, and the scope closes on the way out.
    m.withProvenance(1, () => {
        m.provenanceSetSource('in.msh', 'gmsh');
        m.provenanceNote('regions-dropped', 'Side regions have no OBJ equivalent');
        m.writeMesh('/on.obj', mesh, 'obj');
    });
    const on = m.readProvenance('/on.obj');
    assert(on.recognised, 'scoped write should be recognised');
    assert(on.lines.some((l) => l.includes('in.msh')), 'source missing from block');
    assert(
        on.lines.some((l) => l.includes('regions-dropped')),
        'note missing from block',
    );

    // The scope must not have survived the callback.
    m.writeMesh('/after.obj', mesh, 'obj');
    assert(
        m.readProvenance('/after.obj').lines.length === 1,
        'the scope leaked past withProvenance',
    );

    // A throw inside the body still closes the scope -- the whole reason
    // withProvenance exists rather than raw begin/end.
    try {
        m.withProvenance(1, () => {
            throw new Error('boom');
        });
    } catch (e) {
        if (e.message !== 'boom') throw e;
    }
    m.writeMesh('/after2.obj', mesh, 'obj');
    assert(
        m.readProvenance('/after2.obj').lines.length === 1,
        'a throw inside withProvenance stranded the scope',
    );
});

step('every binding is reachable through the wrapper', () => {
    // Regression guard for the v7.2.1 class of bug: every binding must be
    // forwarded by src/index.mjs, not merely bound in js_bindings.cpp.
    // This list is exhaustive on purpose -- it used to cover only the geometry
    // ops, which left the I/O and data surfaces unguarded.
    for (const name of [
        'readMesh',
        'readMeshSelective',
        'readMetadata',
        'readerSupportsOptions',
        'writeMesh',
        'convert',
        'convertSurface',
        'convertSurfaceOps',
        'runPipeline',
        'numNodesPerCell',
        'topologicalDimension',
        'availableFormats',
        'withProvenance',
        'provenanceBegin',
        'provenanceEnd',
        'provenanceNote',
        'provenanceSetSource',
        'provenanceSetTarget',
        'readProvenance',
        'dataDrop',
        'dataKeep',
        'dataRename',
        'dataPointToCell',
        'dataCellToPoint',
        'dataCalc',
        'dataCondition',
        'dataInfo',
        'dataIntegrate',
        'extractSurface',
        'extractSkin',
        'attachQuality',
        'sniffFormat',
        'reorder',
        'computeBandwidth',
        'diff',
        'meshesEqual',
        'merge',
        'transform',
        'clean',
        'smooth',
        'interpolate',
        'conservativeInterpolate',
        'undoGreen',
        'slice',
        'isosurface',
        'gradient',
        'hessian',
        'estimateError',
        'cropBbox',
        'cropPlane',
        'cropPredicate',
        'split',
        'convertCells',
        'subdivide',
        'agglomerate',
        'refine',
        'decimate',
        'remesh',
        'remeshVolume',
        'optimizeVolume',
        'partition',
        'partitionLabels',
        // Regular grids and signed distance. `grid` is the only binding here
        // that takes no input mesh: it creates one.
        'grid',
        'voxelize',
        'surfaceWatertightCheck',
        'sampleDistance',
        'distanceToSurface',
        'computeSdf',
        'computeCurvature',
        'stats',
        'meshBackend',
        'hasCgnslib',
        'parallelBackend',
        // The transient-XDMF surface is a handle, so the wrapper forwards one
        // factory rather than the seven raw xdmfSeries* bindings; the series
        // steps below assert the handle's own methods, which is where a
        // missing raw forward would show up.
        'createXdmfTimeSeriesWriter',
        // Sequences (multi-file / transient datasets) over MEMFS paths.
        'sequenceEntries',
        'sequenceToTimeseries',
        'timeseriesToSequence',
    ]) {
        assert.equal(typeof m[name], 'function', `${name} is not forwarded by the wrapper`);
    }
});

step('a forwarded geometry operation actually runs', () => {
    const skin = m.extractSurface(cube);
    assert.equal(skin.cells[0].type, 'quad');
    assert.equal(skin.cells[0].data.length, 6 * 4);

    const s = m.stats(cube);
    assert.equal(s.numPoints, 8);
    assert.equal(s.numInverted, 0);

    const pieces = m.split(cube, 'type');
    assert.equal(pieces.length, 1);
    assert.equal(pieces[0].key, 'hexahedron');

    assert.equal(m.meshesEqual(cube, cube), true);
    assert.equal(typeof m.meshBackend(), 'string');
});

step('availableFormats reports what this build can read and write', () => {
    const { readers, writers } = m.availableFormats();
    assert.ok(Array.isArray(readers) && Array.isArray(writers));
    assert.ok(readers.includes('vtu') && writers.includes('vtu'));
    // The two lists genuinely differ: svg/tikz are write-only. A viewer that
    // assumes one list would offer broken menu items.
    assert.ok(writers.includes('svg') && !readers.includes('svg'));
    // openfoam was read-only until v9.20.0 -- this step used to assert that.
    assert.ok(readers.includes('openfoam') && writers.includes('openfoam'));
    // The HDF5- and netCDF-backed formats are in this build too (the wasm32
    // libhdf5/libnetcdf come from build/build-wasm-deps.sh) -- if any of these
    // is missing, the artifact was linked without its dependency and the
    // regression is silent everywhere else, since the registry simply omits
    // the entry rather than failing.
    for (const fmt of ['cgns', 'h5m', 'hmf', 'med', 'exodus'])
        assert.ok(readers.includes(fmt) && writers.includes(fmt), `missing format: ${fmt}`);
    // Sorted, so a UI can render them without sorting again.
    assert.deepEqual(readers, [...readers].sort());
    assert.deepEqual(writers, [...writers].sort());
    // gmsh22 is a WRITE-ONLY registry key (reading auto-detects the version
    // from the file itself, so there is no separate "gmsh22" reader) --
    // before this entry, WASM could select only the lossy 4.1 writer and had
    // no way to reach the one that round-trips region MEMBERSHIP.
    assert.ok(writers.includes('gmsh22') && !readers.includes('gmsh22'));
    // .vti (VTK XML ImageData), v9.25.0: the one format whose Origin/Spacing/
    // WholeExtent attributes ARE a generated grid's header, so it is the only
    // one that round-trips it. Both directions.
    assert.ok(readers.includes('vti') && writers.includes('vti'));
});

step('.vti round-trips a lattice through MEMFS', () => {
    // The point of the format, over the wrapper rather than through C++: a grid
    // written and read back is the same grid, geometry included.
    const g = m.grid([3, 3, 3], [-0.5, -0.5, -0.5], [0.25, 0.25, 0.25]);
    m.writeMesh('/lattice.vti', g);
    const back = m.readMesh('/lattice.vti');
    assert.equal(back.cells[0].type, 'hexahedron');
    assert.equal(back.cells[0].data.length, 27 * 8);
    assert.equal(back.points.length, g.points.length);
    for (let i = 0; i < g.points.length; ++i)
        assert.ok(Math.abs(back.points[i] - g.points[i]) < 1e-12);
    // A mesh that is not a lattice has no extent to write, and says so.
    assert.throws(() => m.writeMesh('/no.vti', cubeSurface));
});

step('openfoam writes a polyMesh DIRECTORY into MEMFS and reads it back', () => {
    // The only writer that creates a directory rather than a file, so it is
    // the only one whose MEMFS behaviour is not covered by every other step.
    const hex = {
        points: new Float64Array([
            0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
        ]),
        dim: 3,
        cells: [{ type: 'hexahedron', data: new Int32Array([0, 1, 2, 3, 4, 5, 6, 7]), nodesPerCell: 8 }],
    };
    m.writeMesh('/of/case.foam', hex, 'openfoam');
    for (const f of ['points', 'faces', 'owner', 'neighbour', 'boundary'])
        assert.ok(m.FS.readFile(`/of/constant/polyMesh/${f}`).length > 0, `missing ${f}`);

    const back = m.readMesh('/of/case.foam', 'openfoam');
    assert.equal(back.points.length, 24);
    assert.ok(back.cells.some((c) => c.type === 'hexahedron'));
});

step('gmsh22 round-trips a region-only mesh; gmsh (4.1) needs entity structure', () => {
    const tagged = {
        ...tet,
        regions: [{ name: 'solid', kind: 'cell', dim: 3, tag: 7, entries: Int32Array.from([0]) }],
    };
    m.writeMesh('/regions22.msh', tagged, 'gmsh22');
    const back22 = m.readMesh('/regions22.msh', 'gmsh');
    assert.equal(back22.regions.length, 1);
    assert.equal(back22.regions[0].name, 'solid');
    assert.deepEqual(Array.from(back22.regions[0].entries), [0]);

    // 4.1 records membership in $Entities, which describes the *geometry* --
    // so it can only be written for a mesh that says which entity each node
    // belongs to (gmsh:dim_tags). This mesh came from another format and has
    // none, so no $Entities is emitted and only the name survives. A file that
    // does carry the structure round-trips: see the next step.
    m.writeMesh('/regions41.msh', tagged, 'gmsh');
    const back41 = m.readMesh('/regions41.msh', 'gmsh');
    assert.equal(back41.regions.length, 0);
});

step('gmsh 4.1 $Entities: physical groups read, and survive a 4.1 round-trip', () => {
    // A real gmsh 4.1 file: $Entities carries the physical tags, so it is the
    // only thing standing between a mesher's own output and this build. It
    // used to throw outright, which made every 4.1 file unreadable here.
    // The unit square as two triangles; surface 1 -> tag 7 "plate", curve 1 ->
    // tag 8 "bottom", curve 2 deliberately untagged.
    const msh = [
        '$MeshFormat', '4.1 0 8', '$EndMeshFormat',
        '$PhysicalNames', '2', '1 8 "bottom"', '2 7 "plate"', '$EndPhysicalNames',
        '$Entities', '4 2 1 0',
        '1 0 0 0 0', '2 1 0 0 0', '3 1 1 0 0', '4 0 1 0 0',
        '1 0 0 0 1 0 0 1 8 2 1 -2',
        '2 1 0 0 1 1 0 0 2 2 -3',
        '1 0 0 0 1 1 0 1 7 2 1 2',
        '$EndEntities',
        '$Nodes', '3 4 1 4',
        '0 1 0 1', '1', '0 0 0',
        '0 2 0 1', '2', '1 0 0',
        '2 1 0 2', '3', '4', '1 1 0', '0 1 0',
        '$EndNodes',
        '$Elements', '3 4 1 5',
        '1 1 1 1', '1 1 2',
        '1 2 1 1', '2 2 3',
        '2 1 2 2', '4 1 2 3', '5 1 3 4',
        '$EndElements', '',
    ].join('\n');
    m.FS.writeFile('/entities.msh', msh);

    const mesh = m.readMesh('/entities.msh', 'gmsh');
    assert.equal(mesh.points.length / 3, 4);
    assert.deepEqual(mesh.cells.map((c) => c.type), ['line', 'line', 'triangle']);
    const names = mesh.regions.map((r) => r.name).sort();
    assert.deepEqual(names, ['bottom', 'plate']);
    const plate = mesh.regions.find((r) => r.name === 'plate');
    assert.equal(plate.dim, 2);
    assert.equal(plate.tag, 7);
    // Global block-major cell indices: the two lines are 0 and 1, so the
    // surface's two triangles are 2 and 3.
    assert.deepEqual(Array.from(plate.entries), [2, 3]);

    // And back out as 4.1: the entity structure rides along, so membership
    // survives -- which before $Entities was written only gmsh22 could do.
    m.writeMesh('/entities-rt.msh', mesh, 'gmsh');
    const back = m.readMesh('/entities-rt.msh', 'gmsh');
    assert.deepEqual(back.regions.map((r) => r.name).sort(), ['bottom', 'plate']);
    const plateBack = back.regions.find((r) => r.name === 'plate');
    assert.equal(plateBack.dim, 2);
    assert.equal(plateBack.tag, 7);
    assert.deepEqual(Array.from(plateBack.entries), [2, 3]);
});

step('gmsh 4.1 with physical groups converts straight to MED (no Python fallback)', () => {
    // The exact downstream repro (CAD-Preview's Gmsh -> MED bridge): a real
    // MSH 4.1 file with $PhysicalNames, one cell block per *entity* (so two
    // "line" blocks here -- MED's own same-type restriction, which the
    // gmsh:physical throw used to make unreachable together), converted
    // straight through this build with no Python anywhere to fall back to.
    const msh = [
        '$MeshFormat', '4.1 0 8', '$EndMeshFormat',
        '$PhysicalNames', '2', '1 8 "bottom"', '2 7 "plate"', '$EndPhysicalNames',
        '$Entities', '4 2 1 0',
        '1 0 0 0 0', '2 1 0 0 0', '3 1 1 0 0', '4 0 1 0 0',
        '1 0 0 0 1 0 0 1 8 2 1 -2',
        '2 1 0 0 1 1 0 0 2 2 -3',
        '1 0 0 0 1 1 0 1 7 2 1 2',
        '$EndEntities',
        '$Nodes', '3 4 1 4',
        '0 1 0 1', '1', '0 0 0',
        '0 2 0 1', '2', '1 0 0',
        '2 1 0 2', '3', '4', '1 1 0', '0 1 0',
        '$EndNodes',
        '$Elements', '3 4 1 5',
        '1 1 1 1', '1 1 2',
        '1 2 1 1', '2 2 3',
        '2 1 2 2', '4 1 2 3', '5 1 3 4',
        '$EndElements', '',
    ].join('\n');
    m.FS.writeFile('/plate.msh', msh);

    // No throw: this used to be "MED: gmsh physical groups handled by
    // Python fallback", fatal with no Python anywhere in this build.
    m.convert('/plate.msh', '/plate.med');
    assert.ok(m.FS.stat('/plate.med').size > 0, '/plate.med is empty');

    const back = m.readMesh('/plate.med', 'med');
    // Both "line" entities consolidated into ONE MED section rather than
    // throwing "MED files cannot have two sections of the same cell type" --
    // i.e. exactly two blocks survive (line, triangle), not three.
    assert.deepEqual(back.cells.map((c) => c.type).sort(), ['line', 'triangle']);
    const names = back.regions.map((r) => r.name).sort();
    assert.deepEqual(names, ['bottom', 'plate']);
});

step('MED consolidates same-type blocks instead of rejecting them', () => {
    // Isolates gap 2 from gap 1: two "triangle" blocks with no gmsh
    // involvement at all, which used to throw
    // "MED files cannot have two sections of the same cell type." up front.
    const mesh = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0]),
        dim: 3,
        cells: [
            { type: 'triangle', data: new Int32Array([0, 1, 2]), nodesPerCell: 3 },
            { type: 'triangle', data: new Int32Array([0, 2, 3, 1, 4, 2]), nodesPerCell: 3 },
        ],
    };
    m.writeMesh('/two-tri.med', mesh, 'med');
    const back = m.readMesh('/two-tri.med', 'med');
    assert.equal(back.cells.length, 1);
    assert.equal(back.cells[0].type, 'triangle');
    assert.equal(back.cells[0].data.length, 9);  // 3 triangles total, consolidated
});

step('CGNS round-trips a surface-only (triangle) mesh', () => {
    // Gap 3's exact repro: the pre-v9.8.0 writer only ever emitted the
    // FIRST "tetra" block, so a triangle-only mesh wrote a file with empty
    // ElementRange/ElementConnectivity groups -- readable by nothing,
    // including this build's own reader.
    m.writeMesh('/tri.cgns', tri2, 'cgns');
    assert.ok(m.FS.stat('/tri.cgns').size > 0, '/tri.cgns is empty');
    const back = m.readMesh('/tri.cgns', 'cgns');
    assert.equal(back.cells.length, 1);
    assert.equal(back.cells[0].type, 'triangle');
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2, 0, 2, 3]);
});

// --- multi-component (vector/tensor) data across the object boundary --------
//
// Before v9.9.0 point_data/cell_data/field_data crossed as flat, SHAPELESS
// Float64Arrays: an (n,3) vector field re-entered C++ as (3n,1), so MED refused
// to read its own output ("field data size does not match its declared shape")
// and every operation silently passed the array through untouched instead of
// gathering it. The fix is the sibling `*_components` objects, which
// `mesh_to_val` now emits and `val_to_mesh` now honours. Note these steps go
// through the OBJECT entry points (writeMesh/readMesh/refine) on purpose --
// convertSurfaceOps is path-in/path-out and never crossed the boundary at all,
// which is why its own multi-component step passed even while this was broken.

const vecMesh = {
    points: new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
    dim: 3,
    cells: [{ type: 'triangle', data: new Int32Array([0, 1, 2, 0, 2, 3]), nodesPerCell: 3 }],
    point_data: {
        temperature: new Float64Array([1, 2, 3, 4]),
        // 4 points x 3 components, interleaved.
        velocity: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    },
    point_data_components: { velocity: 3 },
    cell_data: {
        // 2 cells x 6 components.
        stress: [new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])],
    },
    cell_data_components: { stress: 6 },
};

for (const [format, path] of [['med', '/vec.med'], ['vtu', '/vec.vtu']]) {
    step(`${format}: a vector field survives a JS -> write -> read -> JS round trip`, () => {
        m.writeMesh(path, vecMesh, format);
        assert.ok(m.FS.stat(path).size > 0, `${path} is empty`);
        const back = m.readMesh(path, format);

        // The component counts come back, so the shape is not merely correct by
        // accident of length -- a consumer can reconstruct the (n,3) view.
        assert.equal(back.point_data_components.velocity, 3, 'velocity components');
        assert.equal(back.cell_data_components.stress, 6, 'stress components');
        // A scalar gets no entry at all: absent means one component.
        assert.equal(back.point_data_components.temperature, undefined);

        assert.deepEqual(
            Array.from(back.point_data.velocity),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        );
        assert.deepEqual(Array.from(back.point_data.temperature), [1, 2, 3, 4]);
        assert.deepEqual(
            Array.from(back.cell_data.stress[0]),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        );
    });
}

step('a vector field survives an operation that changes the point count', () => {
    // The silently-broken case: with a (3n,) array the operation's
    // rows == num_points test failed, so it took the pass-through branch and
    // returned the ORIGINAL 12 values against a refined point count.
    const refined = m.refine(vecMesh, 1);
    const nPoints = refined.points.length / 3;
    assert.ok(nPoints > 4, `expected more than 4 points, got ${nPoints}`);
    assert.equal(refined.point_data_components.velocity, 3);
    // One value per point per component -- interpolated, so the length is what
    // proves the gather ran rather than the array being handed back untouched.
    assert.equal(refined.point_data.velocity.length, nPoints * 3);
    // The four original points keep their exact values (refine appends).
    assert.deepEqual(
        Array.from(refined.point_data.velocity.slice(0, 12)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    );
});

step('a bad component count throws a catchable Error', () => {
    const bad = {
        ...vecMesh,
        point_data_components: { velocity: 5 },  // 12 is not a multiple of 5
    };
    assert.throws(
        () => m.writeMesh('/bad-components.vtu', bad, 'vtu'),
        (err) => err instanceof Error && /components/.test(err.message),
    );
    assert.throws(
        () => m.writeMesh('/bad-components.vtu', { ...vecMesh, point_data_components: { velocity: 0 } }, 'vtu'),
        (err) => err instanceof Error,
    );
});

step('convertSurface turns a volume mesh into its renderable boundary', () => {
    m.writeMesh('/cube.vtu', cube);
    m.convertSurface('/cube.vtu', '/cube-surf.vtp');
    const surf = m.readMesh('/cube-surf.vtp');
    // One hexahedron -> 6 boundary quads, and no 3D cells left.
    assert.equal(surf.cells.length, 1);
    assert.equal(surf.cells[0].data.length, 6 * 4);
    assert.equal(surf.points.length, 8 * 3);
});

step('convertSurface carries cell data onto the boundary facets', () => {
    // extract_surface drops cell data (a facet is not a cell), but for a
    // viewer the useful answer is the owning cell's value -- colouring a solid
    // by its material tag is the common case, and without the parent-id gather
    // the array would simply vanish on the way to the renderer.
    const tagged = { ...cube, cell_data: { material: [new Float64Array([7])] } };
    m.writeMesh('/tagged.vtu', tagged);
    m.convertSurface('/tagged.vtu', '/tagged-surf.vtp');
    const surf = m.readMesh('/tagged-surf.vtp');
    assert.deepEqual(Array.from(surf.cell_data.material[0]), [7, 7, 7, 7, 7, 7]);
    // The provenance array is plumbing; it must not clutter a colour-by menu.
    assert.equal(surf.cell_data['surface:parent_cell'], undefined);
});

step('convertSurface passes a surface mesh through, linearized', () => {
    const tri6 = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0]),
        dim: 3,
        cells: [{ type: 'triangle6', data: new Int32Array([0, 1, 2, 3, 4, 5]), nodesPerCell: 6 }],
        point_data: {},
        cell_data: {},
        field_data: {},
    };
    m.writeMesh('/tri6.vtu', tri6);
    m.convertSurface('/tri6.vtu', '/tri6-surf.vtp');
    const out = m.readMesh('/tri6-surf.vtp');
    // Linearized to a 3-node triangle: a renderer has no mid-side nodes, and
    // drawing the 6-node connectivity verbatim would be visible garbage.
    assert.equal(out.cells[0].data.length, 3);
});

step('convertSurface keeps data the flat JS mesh would have dropped', () => {
    // The reason this binding exists rather than readMesh -> extractSkin ->
    // writeMesh: a 3-component array cannot survive the JS mesh boundary.
    const vtu = `<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
 <UnstructuredGrid><Piece NumberOfPoints="4" NumberOfCells="1">
  <Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">
   0 0 0  1 0 0  0 1 0  0 0 1</DataArray></Points>
  <PointData>
   <DataArray type="Float64" Name="disp" NumberOfComponents="3" format="ascii">
    1 2 3  4 5 6  7 8 9  10 11 12</DataArray>
  </PointData>
  <Cells>
   <DataArray type="Int64" Name="connectivity" format="ascii">0 1 2 3</DataArray>
   <DataArray type="Int64" Name="offsets" format="ascii">4</DataArray>
   <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
  </Cells>
 </Piece></UnstructuredGrid>
</VTKFile>`;
    m.FS.writeFile('/vec.vtu', vtu);
    m.convertSurface('/vec.vtu', '/vec-surf.vtp');
    const text = new TextDecoder().decode(m.FS.readFile('/vec-surf.vtp'));
    assert.match(text, /Name="disp"/);
    assert.match(text, /NumberOfComponents="3"/);
});

step('convertSurfaceOps with an empty pipeline is exactly convertSurface', () => {
    // The viewer calls this for both the plain and the post-operation display,
    // so if the two ever diverged, applying and then undoing an operation
    // would silently change the picture.
    m.writeMesh('/eq.vtu', cube);
    m.convertSurface('/eq.vtu', '/eq-a.vtp');
    const report = m.convertSurfaceOps('/eq.vtu', '/eq-b.vtp', []);
    assert.deepEqual(
        Array.from(m.FS.readFile('/eq-a.vtp')),
        Array.from(m.FS.readFile('/eq-b.vtp'))
    );
    assert.deepEqual(report.steps, []);
    assert.deepEqual(report.warnings, []);
});

step('convertSurfaceOps runs each operation and reports its counters', () => {
    m.writeMesh('/ops.vtu', cube);

    const q = m.convertSurfaceOps('/ops.vtu', '/ops-q.vtp', [{ op: 'quality' }]);
    assert.equal(q.steps[0].op, 'quality');
    assert.ok('quality:scaled_jacobian' in m.readMesh('/ops-q.vtp').cell_data);

    const c = m.convertSurfaceOps('/ops.vtu', '/ops-c.vtp', [{ op: 'clean', weld: true }]);
    assert.equal(typeof c.steps[0].pointsWelded, 'number');

    const s = m.convertSurfaceOps('/ops.vtu', '/ops-s.vtp', [
        { op: 'smooth', method: 'laplacian', iterations: 3, fixBoundary: false },
    ]);
    assert.equal(typeof s.steps[0].numNodesMoved, 'number');

    const r = m.convertSurfaceOps('/ops.vtu', '/ops-r.vtp', [{ op: 'refine', levels: 1 }]);
    assert.equal(r.steps[0].op, 'refine');
    // One hexahedron refines to 8, whose boundary is 6*4 = 24 quads.
    assert.equal(m.readMesh('/ops-r.vtp').cells[0].data.length, 24 * 4);

    const p = m.convertSurfaceOps('/ops.vtu', '/ops-p.vtp', [{ op: 'partition', nparts: 2 }]);
    assert.ok('partition:part' in m.readMesh('/ops-p.vtp').cell_data);

    // Operations compose, in order.
    const both = m.convertSurfaceOps('/ops.vtu', '/ops-both.vtp', [
        { op: 'refine', levels: 1 },
        { op: 'quality' },
    ]);
    assert.deepEqual(both.steps.map((x) => x.op), ['refine', 'quality']);
});

step('runPipeline runs a settings document (object, text, and MEMFS path)', () => {
    // The PascalCase settings vocabulary; convertSurfaceOps' camelCase op
    // specs above dispatch through the same core engine, so the two cannot
    // drift -- this step exercises the settings spelling and the wrapper's
    // three input forms.
    m.writeMesh('/pipe.vtu', cube);
    const settings = {
        Version: 1,
        Input: { Path: '/pipe.vtu' },
        Operations: [{ Op: 'ConvertCells', Mode: 'simplexify' }, { Op: 'Quality' }],
        Output: { Path: '/pipe-out.vtu' },
    };
    const rep = m.runPipeline(settings);
    assert.deepEqual(rep.steps.map((x) => x.op), ['ConvertCells', 'Quality']);
    assert.deepEqual(rep.warnings, []);
    const out = m.readMesh('/pipe-out.vtu');
    assert.equal(out.cells[0].type, 'tetra');
    assert.ok('quality:scaled_jacobian' in out.cell_data);

    // JSON text and a MEMFS settings-file path resolve to the same run.
    settings.Output.Path = '/pipe-out2.vtu';
    m.runPipeline(JSON.stringify(settings));
    assert.equal(m.readMesh('/pipe-out2.vtu').cells[0].type, 'tetra');

    settings.Output.Path = '/pipe-out3.vtu';
    m.FS.writeFile('/pipe.json', JSON.stringify(settings));
    m.runPipeline('/pipe.json');
    assert.equal(m.readMesh('/pipe-out3.vtu').cells[0].type, 'tetra');

    // Strict: an unknown op or key fails by name, and excluded multi-mesh
    // ops point at the CLI verb.
    assert.throws(
        () => m.runPipeline({ Input: { Path: '/pipe.vtu' }, Operations: [{ Op: 'Nope' }], Output: { Path: '/x.vtu' } }),
        /unknown operation 'Nope'/
    );
    assert.throws(
        () => m.runPipeline({ Input: { Path: '/pipe.vtu' }, Operations: [], Output: { Path: '/x.vtu' }, Bogus: 1 }),
        /unknown key 'Bogus'/
    );
    assert.throws(
        () => m.runPipeline({ Input: { Path: '/pipe.vtu' }, Operations: [{ Op: 'Merge' }], Output: { Path: '/x.vtu' } }),
        /CLI verb/
    );
});

step('sequences: natural ordering, fan-in, fan-out over MEMFS', () => {
    // A set of MEMFS files treated as one transient dataset. Deliberately
    // UNPADDED names, so the ordering rule has something to prove: a plain
    // sort would put out_10 third.
    m.FS.mkdir('/seq');
    for (let i = 0; i < 12; ++i) m.writeMesh(`/seq/out_${i}.vtu`, cube);

    const plan = m.sequenceEntries('/seq/out_*.vtu');
    assert.equal(plan.length, 12);
    assert.deepEqual(
        plan.map((e) => e.path.split('/').pop()),
        Array.from({ length: 12 }, (_, i) => `out_${i}.vtu`),
        'sequence entries must be in natural-numeric order (out_9 before out_10)'
    );
    // Times come from each filename's trailing digit run, and the entry says so.
    assert.deepEqual(plan.map((e) => e.time), Array.from({ length: 12 }, (_, i) => i));
    assert.equal(plan[0].timeSource, 'filename');
    assert.equal(plan[0].step, 0);

    // Fan-in: 12 single-step files -> one multi-step XDMF.
    const written = m.sequenceToTimeseries('/seq/out_*.vtu', '/seq/series.xdmf');
    assert.equal(written, 12);
    const series = m.sequenceEntries('/seq/series.xdmf');
    assert.equal(series.length, 12, 'the series must report its own 12 steps');
    assert.equal(series[3].timeSource, 'file', 'a series knows its own step times');
    assert.equal(series[3].step, 3);

    // Fan-out: back to one file per step, {step} zero-padded to four digits.
    const paths = m.timeseriesToSequence('/seq/series.xdmf', '/seq/back_{step}.vtu');
    assert.equal(paths.length, 12);
    assert.equal(paths[0], '/seq/back_0000.vtu');
    assert.equal(paths[11], '/seq/back_0011.vtu');
    const back = m.readMesh('/seq/back_0011.vtu');
    assert.equal(back.points.length, cube.points.length);

    // An explicit path list is a stated order and is NOT re-sorted.
    const listed = m.sequenceEntries(['/seq/out_2.vtu', '/seq/out_0.vtu']);
    assert.equal(listed[0].path, '/seq/out_2.vtu');
    assert.equal(m.sequenceEntries(['/seq/out_2.vtu', '/seq/out_0.vtu'], { sort: true })[0].path,
                 '/seq/out_0.vtu');

    // Explicit times win, and report themselves as such.
    const timed = m.sequenceEntries('/seq/out_*.vtu',
                                    { times: Array.from({ length: 12 }, (_, i) => i * 0.25) });
    assert.equal(timed[4].time, 1.0);
    assert.equal(timed[4].timeSource, 'explicit');
});

step('sequences fail by name rather than truncating', () => {
    // A format that cannot hold a series names itself and the remedy.
    assert.throws(
        () => m.sequenceToTimeseries('/seq/out_*.vtu', '/seq/bad.vtu'),
        /cannot hold a multi-step series/
    );
    // A fan-out needs a {step}/{index} token.
    assert.throws(
        () => m.timeseriesToSequence('/seq/series.xdmf', '/seq/plain.vtu'),
        /\{step\}/
    );
    // A pattern matching nothing is an error, never an empty sequence.
    assert.throws(() => m.sequenceEntries('/seq/nothing_*.vtu'), /matched no files/);
    // The directory component of a pattern is taken literally.
    assert.throws(() => m.sequenceEntries('/se*/out_*.vtu'), /taken literally/);
    // A mistyped option is named.
    assert.throws(() => m.sequenceEntries('/seq/out_*.vtu', { timeFrom: 'vibes' }), /TimeFrom/);
    assert.throws(() => m.sequenceEntries('/seq/out_*.vtu', { bogus: 1 }), /unknown key 'bogus'/);
});

step('runPipeline runs a whole transient dataset per step', () => {
    // The composition that makes the pipeline a batch post-processor: a
    // Pattern input and a {step} output route the SAME document to the
    // sequence driver, with the chain applied to every step.
    const rep = m.runPipeline({
        Version: 1,
        Input: { Pattern: '/seq/out_*.vtu' },
        Operations: [{ Op: 'Quality' }],
        Output: { Path: '/seq/post_{step}.vtu' },
    });
    assert.equal(rep.steps.length, 12, 'one report entry per (step, op)');
    assert.deepEqual(new Set(rep.steps.map((x) => x.op)), new Set(['Quality']));
    const post = m.readMesh('/seq/post_0007.vtu');
    assert.ok('quality:scaled_jacobian' in post.cell_data);

    // Mode ASSERTS the inferred shape rather than selecting it.
    assert.throws(
        () => m.runPipeline({
            Version: 1,
            Mode: 'fan-in',
            Input: { Paths: ['/seq/out_0.vtu'] },
            Output: { Path: '/seq/one.xdmf' },
        }),
        /Mode says 'fan-in'/
    );
    // A multi-step input aimed at a single-step output refuses to truncate.
    assert.throws(
        () => m.runPipeline({
            Version: 1,
            Input: { Path: '/seq/series.xdmf' },
            Output: { Path: '/seq/trunc.vtu' },
        }),
        /12 time steps/
    );
});

step('convertSurfaceOps takes a planar cross-section (slice)', () => {
    m.writeMesh('/sec.vtu', cube);
    // The mid-plane cross-section of the unit cube is a non-empty surface (a
    // unit square of section faces at z = 0.5), rendered directly.
    const r = m.convertSurfaceOps('/sec.vtu', '/sec.vtp', [
        { op: 'section', point: [0.5, 0.5, 0.5], normal: [0, 0, 1] },
    ]);
    assert.equal(r.steps[0].op, 'section');
    assert.equal(r.warnings.length, 0, 'a plane through the mesh must not warn');
    const sec = m.readMesh('/sec.vtp');
    assert.ok(sec.cells.length > 0, 'the section is non-empty');
    for (const block of sec.cells)
        for (let i = 2; i < sec.points.length; i += 3)
            assert.ok(Math.abs(sec.points[i] - 0.5) < 1e-9, 'section lies at z = 0.5');

    // A plane that misses the mesh yields an empty section, and that is warned.
    const miss = m.convertSurfaceOps('/sec.vtu', '/sec-miss.vtp', [
        { op: 'section', point: [0.5, 0.5, 5], normal: [0, 0, 1] },
    ]);
    assert.ok(miss.warnings.length > 0, 'an empty section must warn');
});

step('convertSurfaceOps keeps multi-component data through an operation', () => {
    // The whole reason this binding exists: the same pipeline expressed as
    // readMesh -> smooth -> writeMesh would flatten `disp` on the way through.
    const vtu = `<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
 <UnstructuredGrid><Piece NumberOfPoints="4" NumberOfCells="1">
  <Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">
   0 0 0  1 0 0  0 1 0  0 0 1</DataArray></Points>
  <PointData>
   <DataArray type="Float64" Name="disp" NumberOfComponents="3" format="ascii">
    1 2 3  4 5 6  7 8 9  10 11 12</DataArray>
  </PointData>
  <Cells>
   <DataArray type="Int64" Name="connectivity" format="ascii">0 1 2 3</DataArray>
   <DataArray type="Int64" Name="offsets" format="ascii">4</DataArray>
   <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
  </Cells>
 </Piece></UnstructuredGrid>
</VTKFile>`;
    m.FS.writeFile('/vec-ops.vtu', vtu);
    m.convertSurfaceOps('/vec-ops.vtu', '/vec-ops.vtp', [{ op: 'quality' }]);
    const text = new TextDecoder().decode(m.FS.readFile('/vec-ops.vtp'));
    assert.match(text, /Name="disp"/);
    assert.match(text, /NumberOfComponents="3"/);
});

step('convertSurfaceOps can keep the provenance array for a picker', () => {
    m.writeMesh('/prov.vtu', cube);
    m.convertSurfaceOps('/prov.vtu', '/prov.vtp', [], { keepProvenance: true });
    assert.ok('surface:parent_cell' in m.readMesh('/prov.vtp').cell_data);
    m.convertSurfaceOps('/prov.vtu', '/prov2.vtp', []);
    assert.equal(m.readMesh('/prov2.vtp').cell_data['surface:parent_cell'], undefined);
});

step('convertSurfaceOps rejects an unknown operation by name', () => {
    m.writeMesh('/bad.vtu', cube);
    assert.throws(
        () => m.convertSurfaceOps('/bad.vtu', '/bad.vtp', [{ op: 'teleport' }]),
        /unknown operation 'teleport'/
    );
});

step('named regions round-trip through the wrapper', () => {
    // Regions ride on the mesh object rather than through a callable, so this
    // is their equivalent of the exhaustive forward guard above: if
    // `mesh_to_val` / `val_to_mesh` stopped carrying them, this fails.
    // (That is also why the list above did not grow -- nothing new is
    // forwarded by src/index.mjs; see doc/regions.md.)
    const tagged = {
        ...tet,
        regions: [
            { name: 'fixed', kind: 'point', dim: -1, tag: -1, entries: Int32Array.from([0, 3]) },
            { name: 'solid', kind: 'cell', dim: 3, tag: 42, entries: Int32Array.from([0]) },
            { name: 'wall', kind: 'side', dim: 2, tag: -1, entries: Int32Array.from([0, 1]) },
        ],
    };
    m.writeMesh('/regions.inp', tagged);
    const back = m.readMesh('/regions.inp');
    assert.ok(Array.isArray(back.regions), 'regions is an array');

    const byName = Object.fromEntries(back.regions.map((r) => [r.name, r]));
    assert.deepEqual(Object.keys(byName).sort(), ['fixed', 'solid', 'wall']);
    assert.equal(byName.fixed.kind, 'point');
    assert.deepEqual(Array.from(byName.fixed.entries), [0, 3]);
    assert.equal(byName.solid.kind, 'cell');
    assert.equal(byName.wall.kind, 'side');
    // A side region's entries are (cell, facet) pairs, so two values per entry.
    assert.equal(byName.wall.entries.length % 2, 0);
    assert.deepEqual(Array.from(byName.wall.entries), [0, 1]);
});

step('regions survive an operation', () => {
    const tagged = {
        ...tet,
        regions: [
            { name: 'solid', kind: 'cell', dim: 3, tag: 7, entries: Int32Array.from([0]) },
        ],
    };
    const out = m.cropBbox(tagged, [-9, -9, -9], [9, 9, 9], 'all', false);
    const solid = out.regions.find((r) => r.name === 'solid');
    assert.ok(solid, 'the region survived the crop');
    assert.equal(solid.tag, 7, 'the format-native id rides along');
});

step('sniffFormat identifies a file by its leading bytes', () => {
    // Deliberately misleading extension: sniffing must go by content.
    m.writeMesh('/sniffme.vtu', tet);
    m.FS.writeFile('/sniffme.dat', m.FS.readFile('/sniffme.vtu'));
    assert.equal(m.sniffFormat('/sniffme.dat'), 'vtu');
    // Only a confident signature match is claimed; anything else returns "".
    m.FS.writeFile('/ambiguous.dat', 'nothing recognizable here\n');
    assert.equal(m.sniffFormat('/ambiguous.dat'), '');
});

step('ragged (polygon) cell blocks cross the JS boundary as CSR arrays', () => {
    // Previously any ragged block threw unconditionally ("not supported by
    // the JS API yet") on BOTH read and write. Represented as two flat CSR
    // arrays -- `data` (every row's node ids concatenated) and `rowOffsets`
    // (each cell's start index into `data`, length numCells + 1) -- since
    // embind has no efficient representation for a nested array of arrays.
    const poly = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 1, 0]),
        dim: 3,
        cells: [
            {
                type: 'polygon',
                data: new Int32Array([0, 1, 2, 1, 3, 4, 2]), // a triangle then a 4-gon
                rowOffsets: new Int32Array([0, 3, 7]),
            },
        ],
    };
    // MED is the C++ core's ragged-polygon-capable writer (POG/POG2).
    m.writeMesh('/ragged.med', poly, 'med');
    const back = m.readMesh('/ragged.med', 'med');
    assert.equal(back.cells.length, 1);
    assert.equal(back.cells[0].type, 'polygon');
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2, 1, 3, 4, 2]);
    assert.deepEqual(Array.from(back.cells[0].rowOffsets), [0, 3, 7]);
});

step('ragged (polyhedron) cell blocks cross the JS boundary as CSR arrays', () => {
    // 2-level ragged (cell -> faces -> node ids), as three flat CSR arrays:
    // `data`, `faceOffsets` (per-face start into `data`), `cellOffsets`
    // (per-cell start into the face list). No C++ format writer accepts a
    // polyhedron block yet (a documented, pre-existing gap, not new), so this
    // exercises the boundary itself via an operation instead of a file:
    // val_to_mesh -> clean() (a no-op with every flag off) -> mesh_to_val
    // must reproduce the exact input.
    const tetra = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
        dim: 3,
        cells: [
            {
                type: 'polyhedron',
                data: new Int32Array([0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0]), // 4 triangular faces
                faceOffsets: new Int32Array([0, 3, 6, 9, 12]),
                cellOffsets: new Int32Array([0, 4]),
            },
        ],
    };
    const result = m.clean(tetra, false, 0.0, false, false, false);
    const cb = result.mesh.cells[0];
    assert.equal(cb.type, 'polyhedron');
    assert.deepEqual(Array.from(cb.data), Array.from(tetra.cells[0].data));
    assert.deepEqual(Array.from(cb.faceOffsets), Array.from(tetra.cells[0].faceOffsets));
    assert.deepEqual(Array.from(cb.cellOffsets), Array.from(tetra.cells[0].cellOffsets));

    // MED gained MED_POLYHEDRON (POE) in v9.19.0, so this now ROUND-TRIPS
    // rather than throwing -- which is what this step used to assert.
    m.writeMesh('/polyhedron.med', tetra, 'med');
    const back = m.readMesh('/polyhedron.med', 'med');
    assert.equal(back.cells.length, 1);
    assert.ok(back.cells[0].cellOffsets, 'a polyhedron block must come back 2-level');
    assert.equal(back.cells[0].cellOffsets.length, 2);
    assert.equal(back.cells[0].faceOffsets.length, 5); // 4 faces + 1

    // A format that genuinely cannot hold a polyhedron must still fail cleanly
    // -- a catchable Error naming the reason, never a WASM abort. VTP is the
    // honest example: PolyData is 2-D by definition.
    assert.throws(
        () => m.writeMesh('/polyhedron.vtp', tetra, 'vtp'),
        (err) => err instanceof Error && /polyhedron/.test(err.message),
    );
});

step('malformed ragged CSR offsets fail by name, not by reading out of range', () => {
    // val_to_mesh is hostile to caller input by contract: the offsets are the
    // one way a JS caller can steer a read past the end of `data`. The polygon
    // branch always checked this; the polyhedron branch's faceOffsets did not
    // until v9.15.0, which is what this pins.
    const base = {
        points: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
        dim: 3,
    };
    const withCells = (cells) => ({ ...base, cells });

    // faceOffsets running past the end of `data`.
    assert.throws(
        () =>
            m.clean(
                withCells([
                    {
                        type: 'polyhedron',
                        data: new Int32Array([0, 1, 2]),
                        faceOffsets: new Int32Array([0, 99]),
                        cellOffsets: new Int32Array([0, 1]),
                    },
                ]),
                false, 0.0, false, false, false,
            ),
        (err) => err instanceof Error && /faceOffsets/.test(err.message),
    );

    // cellOffsets naming a face the faceOffsets array does not have.
    assert.throws(
        () =>
            m.clean(
                withCells([
                    {
                        type: 'polyhedron',
                        data: new Int32Array([0, 1, 2]),
                        faceOffsets: new Int32Array([0, 3]),
                        cellOffsets: new Int32Array([0, 5]),
                    },
                ]),
                false, 0.0, false, false, false,
            ),
        (err) => err instanceof Error && /cellOffsets/.test(err.message),
    );

    // rowOffsets running past the end of `data`.
    assert.throws(
        () =>
            m.clean(
                withCells([
                    {
                        type: 'polygon',
                        data: new Int32Array([0, 1, 2]),
                        rowOffsets: new Int32Array([0, 42]),
                    },
                ]),
                false, 0.0, false, false, false,
            ),
        (err) => err instanceof Error && /rowOffsets/.test(err.message),
    );
});

// ---------------------------------------------------------------------------
// Transient (time-series) XDMF -- the one STATEFUL binding: an opaque handle
// whose output lands in MEMFS across several calls, rather than a pure
// function over a mesh object. See src/wasm/index.d.ts's XdmfTimeSeriesWriter
// doc comment for why the raw binding is a handle + free functions and not an
// embind class_.
// ---------------------------------------------------------------------------

// A step's mesh: same geometry every time (the writer ignores it after the
// first call), only the field changes.
const seriesStep = (k) => ({
    ...tet,
    point_data: { temperature: new Float64Array([k, k + 1, k + 2, k + 3]) },
    cell_data: { material: [new Float64Array([10 * k])] },
});

step('XDMF time series (HDF): 3 steps, and the .h5 companion is written too', () => {
    const w = m.createXdmfTimeSeriesWriter('/series.xdmf'); // 'HDF' by default
    assert.equal(typeof w.writePointsCells, 'function');
    assert.equal(typeof w.writeData, 'function');
    assert.equal(typeof w.finalize, 'function');
    assert.equal(typeof w.numSteps, 'function');
    assert.equal(typeof w.finalized, 'function');
    assert.equal(typeof w.close, 'function');

    w.writePointsCells(tet);
    assert.equal(w.numSteps(), 0);
    for (let k = 0; k < 3; ++k) w.writeData(k * 0.5, seriesStep(k));
    assert.equal(w.numSteps(), 3);
    assert.equal(w.finalized(), false);
    w.close();

    // TWO files: the light-data .xdmf and its SIBLING heavy-data .h5 (the C++
    // writer resolves it next to the .xdmf, not in the CWD). A consumer that
    // copies only the first one out of MEMFS ships an unreadable series.
    const xdmf = m.FS.readFile('/series.xdmf');
    const h5 = m.FS.readFile('/series.h5');
    assert.ok(xdmf.length > 0, '.xdmf written');
    assert.ok(h5.length > 0, '.h5 companion written');
    assert.match(new TextDecoder().decode(xdmf), /CollectionType="Temporal"/);

    // The wasm build links a wasm32 HDF5 (v8.0.0+), so 'HDF' really is the
    // HDF path -- assert the payload is an actual HDF5 file, not a fallback.
    assert.deepEqual(Array.from(h5.slice(0, 4)), [0x89, 0x48, 0x44, 0x46], 'HDF5 magic');
});

step('XDMF time series (HDF): reads back with the right per-step values', () => {
    // read_xdmf resolves the temporal collection structurally; timeStep picks
    // the step (0 first, negative from the end).
    const meta = m.readMetadata('/series.xdmf');
    assert.deepEqual(Array.from(meta.timeValues), [0, 0.5, 1]);

    for (let k = 0; k < 3; ++k) {
        const back = m.readMeshSelective('/series.xdmf', { timeStep: k });
        assert.equal(back.cells[0].type, 'tetra');
        assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2, 3]);
        assert.deepEqual(Array.from(back.points), Array.from(tet.points));
        assert.deepEqual(Array.from(back.point_data.temperature), [k, k + 1, k + 2, k + 3]);
        assert.deepEqual(Array.from(back.cell_data.material[0]), [10 * k]);
    }
    // -1 is the last step.
    assert.deepEqual(
        Array.from(m.readMeshSelective('/series.xdmf', { timeStep: -1 }).point_data.temperature),
        [2, 3, 4, 5],
    );
});

step('XDMF time series (XML): one self-contained file, no companion', () => {
    const w = m.createXdmfTimeSeriesWriter('/series-xml.xdmf', { dataFormat: 'XML' });
    w.writePointsCells(tet);
    w.writeData(0.0, seriesStep(0));
    w.writeData(1.0, seriesStep(1));
    w.close();

    const text = new TextDecoder().decode(m.FS.readFile('/series-xml.xdmf'));
    assert.match(text, /Format="XML"/);
    assert.throws(() => m.FS.readFile('/series-xml.h5'), 'no HDF companion for XML');

    const back = m.readMeshSelective('/series-xml.xdmf', { timeStep: 1 });
    assert.deepEqual(Array.from(back.point_data.temperature), [1, 2, 3, 4]);
});

step('XDMF time series: flush makes a partial series readable', () => {
    // Without this the light data appears only at finalize, so a run that is
    // killed leaves heavy data and nothing readable.
    const w = m.createXdmfTimeSeriesWriter('/flushed.xdmf', { dataFormat: 'XML' });
    w.writePointsCells(tet);
    w.writeData(0.0, seriesStep(0));
    w.flush();
    assert.deepEqual(Array.from(m.readMetadata('/flushed.xdmf').timeValues), [0]);
    w.writeData(1.0, seriesStep(1));
    w.flush();
    assert.deepEqual(Array.from(m.readMetadata('/flushed.xdmf').timeValues), [0, 1]);
    w.close();
});

step('XDMF time series: append continues an existing collection', () => {
    const w2 = m.createXdmfTimeSeriesWriter('/flushed.xdmf', {
        dataFormat: 'XML',
        mode: 'append',
    });
    assert.equal(w2.numSteps(), 2, 'existing steps are counted');
    w2.writeData(2.0, seriesStep(2));
    w2.close();
    assert.deepEqual(Array.from(m.readMetadata('/flushed.xdmf').timeValues), [0, 1, 2]);
    // Nothing earlier was overwritten.
    assert.deepEqual(
        Array.from(m.readMeshSelective('/flushed.xdmf', { timeStep: 0 }).point_data.temperature),
        [0, 1, 2, 3],
    );
});

step('XDMF time series: writeDataArrays takes raw solver arrays', () => {
    const w = m.createXdmfTimeSeriesWriter('/arrays.xdmf', { dataFormat: 'XML' });
    w.writePointsCells(tet);
    w.writeDataArrays(0.0, { temperature: new Float64Array([1, 2, 3, 4]) });
    w.close();
    const back = m.readMeshSelective('/arrays.xdmf', { timeStep: 0 });
    assert.deepEqual(Array.from(back.point_data.temperature), [1, 2, 3, 4]);
});

step('XDMF time series: nothing is on the FS until the series is finalized', () => {
    // Still true without an explicit flush: the collection element has to
    // enclose every step, so the light data is written at the end by default.
    const w = m.createXdmfTimeSeriesWriter('/pending.xdmf', { dataFormat: 'XML' });
    w.writePointsCells(tet);
    w.writeData(0.0, seriesStep(0));
    assert.throws(() => m.FS.readFile('/pending.xdmf'), 'not written before finalize');
    w.finalize();
    assert.equal(w.finalized(), true);
    assert.ok(m.FS.readFile('/pending.xdmf').length > 0);
    w.close(); // finalize() then close() must not double-write or throw
});

step('XDMF time series: misuse throws readable JS Errors, never a WASM abort', () => {
    assert.throws(
        () => m.createXdmfTimeSeriesWriter('/bad.xdmf', { dataFormat: 'Parquet' }),
        (err) => err instanceof Error && /unknown data format/i.test(err.message),
    );

    const w = m.createXdmfTimeSeriesWriter('/misuse.xdmf', { dataFormat: 'XML' });
    assert.throws(
        () => w.writeData(0.0, seriesStep(0)),
        (err) => err instanceof Error && /WritePointsCells/.test(err.message),
        'writeData before writePointsCells',
    );
    w.writePointsCells(tet);
    assert.throws(
        () => w.writePointsCells(tet),
        (err) => err instanceof Error && /more than once/.test(err.message),
    );
    w.finalize();
    assert.throws(
        () => w.writeData(1.0, seriesStep(0)),
        (err) => err instanceof Error && /finalized/.test(err.message),
    );

    // The handle is a table index, not a pointer: using a closed series is a
    // thrown Error, not a use-after-free in linear memory.
    w.close();
    assert.throws(
        () => w.numSteps(),
        (err) => err instanceof Error && /already-closed|invalid/.test(err.message),
    );
    w.close(); // idempotent -- close() belongs in a `finally`
});

// ---------------------------------------------------------------------------
// The sequential artifact. The suite above ran on the threaded build; this
// block proves the *other* shipped artifact also loads and round-trips, and
// that it really is the sequential one (not a second copy of the mt build).
// Both must be present, since the loader auto-selects between them at runtime.
// ---------------------------------------------------------------------------
const mSeq = await loadMeshioPlusPlus({}, { variant: 'seq' });

step('hasCgnslib reports whether the CGNS MLL is linked into this artifact', () => {
    // Without a probe, a build that silently dropped cgnslib reads every file
    // we produce ourselves identically -- the regression would surface only on
    // a user's ADF-backed file. This artifact is built with it.
    assert.equal(typeof m.hasCgnslib(), 'boolean');
    assert.ok(m.hasCgnslib(), 'this artifact should be linked against cgnslib');
});

step('sequential (seq) build loads and reports the seq parallel backend', () => {
    assert.equal(mSeq.parallelBackend(), 'seq');
});

step('sequential build round-trips a mesh (VTU) and runs an operation', () => {
    mSeq.writeMesh('/seq.vtu', tet);
    const back = mSeq.readMesh('/seq.vtu');
    assert.equal(back.cells[0].type, 'tetra');
    assert.deepEqual(Array.from(back.cells[0].data), [0, 1, 2, 3]);
    assert.deepEqual(Array.from(back.point_data.temperature), [1, 2, 3, 4]);

    // An operation, so the sequential parallel_for paths are exercised too.
    const surf = mSeq.extractSurface(cube);
    assert.equal(surf.cells[0].type, 'quad');
    assert.equal(surf.cells[0].data.length, 6 * 4);
});

if (failed) {
    console.error('\nSMOKE TEST FAILED');
    process.exit(1);
}
console.log('\nSMOKE TEST PASSED');
