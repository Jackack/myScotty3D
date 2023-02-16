#include "pipeline.h"

#include "framebuffer.h"
#include "sample_pattern.h"

#include "../lib/log.h"
#include "../lib/mathlib.h"

#include <iostream>


// opposing directions sum to 0
const int INSIDE_DIAMOND = 0;
const int SW = 1;
const int NE = -1;
const int SE = 2;
const int NW = -2;
const int N = 3;
const int S = -3;
const int E = 4;
const int W = -4;

template< PrimitiveType primitive_type, class Program, uint32_t flags >
void Pipeline< primitive_type, Program, flags >::run(
		std::vector< Vertex > const &vertices,
		typename Program::Parameters const &parameters,
		Framebuffer *framebuffer_) {
	//Framebuffer must be non-null:
	assert(framebuffer_);
	auto &framebuffer = *framebuffer_;

	//A1T7: sample loop
	//TODO: update this function to rasterize to *all* sample locations in the framebuffer.
	// This will probably involve inserting a loop of the form:
	//     std::vector< Vec3 > const &samples = framebuffer.sample_pattern.centers_and_weights;
	//     for (uint32_t s = 0; s < samples.size(); ++s) { ... }
	//  around some subset of the code.
	// You will also need to transform the input and output of the rasterize_* functions to
	//   account for the fact they deal with pixels centered at (0.5,0.5).

	std::vector< ShadedVertex > shaded_vertices;
	shaded_vertices.reserve(vertices.size());

	//--------------------------
	//shade vertices:
	for (auto const &v : vertices) {
		ShadedVertex sv;
		Program::shade_vertex( parameters, v.attributes, &sv.clip_position, &sv.attributes );
		shaded_vertices.emplace_back(sv);
	}

	//--------------------------
	//assemble + clip + homogeneous divide vertices:
	std::vector< ClippedVertex > clipped_vertices;

	//reserve some space to avoid reallocations later:
	if constexpr (primitive_type == PrimitiveType::Lines) {
		//clipping lines can never produce more than one vertex per input vertex:
		clipped_vertices.reserve(shaded_vertices.size());
	} else if constexpr (primitive_type == PrimitiveType::Triangles) {
		//clipping triangles can produce up to 8 vertices per input vertex:
		clipped_vertices.reserve(shaded_vertices.size() * 8);
	}

	//coefficients to map from clip coordinates to framebuffer (i.e., "viewport") coordinates:
	//x: [-1,1] -> [0,width]
	//y: [-1,1] -> [0,height]
	//z: [-1,1] -> [0,1] (OpenGL-style depth range)
	Vec3 const clip_to_fb_scale = Vec3{
		framebuffer.width / 2.0f,
		framebuffer.height / 2.0f,
		0.5f
	};
	Vec3 const clip_to_fb_offset = Vec3{
		0.5f * framebuffer.width,
		0.5f * framebuffer.height,
		0.5f
	};

	//helper used to put output of clipping functions into clipped_vertices:
	auto emit_vertex = [&](ShadedVertex const &sv) {
		ClippedVertex cv;
		float inv_w = 1.0f / sv.clip_position.w;
		cv.fb_position = clip_to_fb_scale * inv_w * sv.clip_position.xyz() + clip_to_fb_offset;
		cv.inv_w = inv_w;
		cv.attributes = sv.attributes;
		clipped_vertices.emplace_back(cv);
	};

	//actually do clipping:
	if constexpr (primitive_type == PrimitiveType::Lines) {
		for (uint32_t i = 0; i + 1 < shaded_vertices.size(); i += 2) {
			clip_line( shaded_vertices[i], shaded_vertices[i+1], emit_vertex );
		}
	} else if constexpr (primitive_type == PrimitiveType::Triangles) {
		for (uint32_t i = 0; i + 2 < shaded_vertices.size(); i += 3) {
			clip_triangle( shaded_vertices[i], shaded_vertices[i+1], shaded_vertices[i+2], emit_vertex );
		}
	} else {
		static_assert( primitive_type == PrimitiveType::Lines, "Unsupported primitive type." );
	}


	//--------------------------
	//rasterize primitives:

	std::vector< Fragment > fragments;

	//helper used to put output of rasterization functions into fragments:
	auto emit_fragment = [&](Fragment const &f) {
		fragments.emplace_back(f);
	};
	//actually do rasterization:
	if constexpr (primitive_type == PrimitiveType::Lines) {
		for (uint32_t i = 0; i + 1 < clipped_vertices.size(); i += 2) {
			rasterize_line( clipped_vertices[i], clipped_vertices[i+1], emit_fragment );
		}
	} else if constexpr (primitive_type == PrimitiveType::Triangles) {
		for (uint32_t i = 0; i + 2 < clipped_vertices.size(); i += 3) {
			rasterize_triangle( clipped_vertices[i], clipped_vertices[i+1], clipped_vertices[i+2], emit_fragment );
		}
	} else {
		static_assert( primitive_type == PrimitiveType::Lines, "Unsupported primitive type." );
	}

	//--------------------------
	//depth test + shade + blend fragments:
	uint32_t out_of_range = 0; //check if rasterization produced fragments outside framebuffer (indicates something is wrong with clipping)
	for (auto const &f : fragments) {

		//fragment location (in pixels):
		int32_t x = (int32_t)std::floor(f.fb_position.x);
		int32_t y = (int32_t)std::floor(f.fb_position.y);

		//if clipping is working properly, this condition shouldn't be needed;
		//however, it prevents crashes while you are working on your clipping functions,
		//so we suggest leaving it in place:
		if (x < 0 || (uint32_t)x >= framebuffer.width || y < 0 || (uint32_t)y >= framebuffer.height) {
			++out_of_range;
			continue;
		}

		//local names that refer to destination sample in framebuffer:
		float &fb_depth = framebuffer.depth_at(x,y,0);
		Spectrum &fb_color = framebuffer.color_at(x,y,0);

		//depth test:
		if constexpr ((flags & PipelineMask_Depth) == Pipeline_Depth_Always) {
			//"Always" means the depth test always passes.
		} else if constexpr ((flags & PipelineMask_Depth) == Pipeline_Depth_Never) {
			//"Never" means the depth test never passes.
			continue; //discard this fragment
		} else if constexpr ((flags & PipelineMask_Depth) == Pipeline_Depth_Less) {
			//"Less" means the depth test passes when the new fragment has depth less than the stored depth.
			//A1T4: Depth_Less
			if (f.fb_position.z >= fb_depth){
				continue;
			}
		} else {
			static_assert((flags & PipelineMask_Depth) <= Pipeline_Depth_Always, "Unknown depth test flag.");
		}

		//if depth test passes, and depth writes aren't disabled, write depth to depth buffer:
		if constexpr (!(flags & Pipeline_DepthWriteDisableBit)) {
			fb_depth = f.fb_position.z;
		}

		//shade fragment:
		ShadedFragment sf;
		sf.fb_position = f.fb_position;
		Program::shade_fragment(parameters, f.attributes, f.derivatives, &sf.color, &sf.opacity);

		//write color to framebuffer if color writes aren't disabled:
		if constexpr (!(flags & Pipeline_ColorWriteDisableBit)) {

			//blend fragment:
			if constexpr ((flags & PipelineMask_Blend) == Pipeline_Blend_Replace) {
				fb_color = sf.color;
			} else if constexpr ((flags & PipelineMask_Blend) == Pipeline_Blend_Add) {
				//A1T4: Blend_Add
				//TODO: framebuffer color should have fragment color multiplied by fragment opacity added to i
				fb_color += (sf.color * sf.opacity);
			} else if constexpr ((flags & PipelineMask_Blend) == Pipeline_Blend_Over) {
				//A1T4: Blend_Over
				//TODO: set framebuffer color to the result of "over" blending (also called "alpha blending") the fragment color over the framebuffer color, using the fragment's opacity
 				// You may assume that the framebuffer color has its alpha premultiplied already, and you just want to compute the resulting composite color
				Spectrum sf_color = sf.opacity * sf.color;
				fb_color = sf_color + (1 - sf.opacity) * fb_color;
			} else {
				static_assert((flags & PipelineMask_Blend) <= Pipeline_Blend_Over, "Unknown blending flag.");
			}
		}
	}

	if (out_of_range > 0) {
		if constexpr (primitive_type == PrimitiveType::Lines) {
			warn("Produced %d fragments outside framebuffer; this indicates something is likely wrong with the clip_line function.", out_of_range);
		} else if constexpr (primitive_type == PrimitiveType::Triangles) {
			warn("Produced %d fragments outside framebuffer; this indicates something is likely wrong with the clip_triangle function.", out_of_range);
		}
	}

	

}

//-------------------------------------------------------------------------
//clipping functions

//helper to interpolate between vertices:
template< PrimitiveType p, class P, uint32_t F >
auto Pipeline< p, P, F >::lerp(ShadedVertex const &a, ShadedVertex const &b, float t) -> ShadedVertex {
	ShadedVertex ret;
	ret.clip_position = (b.clip_position - a.clip_position) * t + a.clip_position;
	for (uint32_t i = 0; i < ret.attributes.size(); ++i) {
		ret.attributes[i] = (b.attributes[i] - a.attributes[i]) * t + a.attributes[i];
	}
	return ret;
}



/*
 * clip_line - clip line to portion with -w <= x,y,z <= w, emit vertices of clipped line (if non-empty)
 *  va, vb: endpoints of line
 *  emit_vertex: call to produce truncated line
 *
 * If clipping shortens the line, attributes of the shortened line should respect the pipeline's interpolation mode.
 * 
 * If no portion of the line remains after clipping, emit_vertex will not be called.
 *
 * The clipped line should have the same direction as the full line.
 *
 */
template< PrimitiveType p, class P, uint32_t flags >
void Pipeline< p, P, flags >::clip_line(
		ShadedVertex const &va, ShadedVertex const &vb,
		std::function< void(ShadedVertex const &) > const &emit_vertex
	) {
	//Determine portion of line over which:
	// pt = (b-a) * t + a
	// -pt.w <= pt.x <= pt.w
	// -pt.w <= pt.y <= pt.w
	// -pt.w <= pt.z <= pt.w

	//... as a range [min_t, max_t]:

	float min_t = 0.0f;
	float max_t = 1.0f;
	
	// want to set range of t for a bunch of equations like:
	//    a.x + t * ba.x <= a.w + t * ba.w
	// so here's a helper:
	auto clip_range = [&min_t, &max_t](float l, float dl, float r, float dr) {
		//restrict range such that:
		//l + t * dl <= r + t * dr
		//re-arranging:
		// l - r <= t * (dr - dl)
		if (dr == dl) {
			//want: l - r <= 0
			if (l - r > 0.0f) {
				//works for none of range, so make range empty:
				min_t = 1.0f; max_t = 0.0f;
			}
		} else if (dr > dl) {
			//since dr - dl is positive:
			//want: (l - r) / (dr - dl) <= t
			min_t = std::max(min_t, (l - r) / (dr - dl));
		} else { //dr < dl
			//since dr - dl is negative:
			//want: (l - r) / (dr - dl) >= t
			max_t = std::min(max_t, (l - r) / (dr - dl));
		}
	};
	
	//local names for clip positions and their difference:
	Vec4 const &a = va.clip_position;
	Vec4 const &b = vb.clip_position;
	Vec4 const ba = b-a;

	// -a.w - t * ba.w <= a.x + t * ba.x <= a.w + t * ba.w
	clip_range(-a.w,-ba.w, a.x, ba.x);
	clip_range( a.x, ba.x, a.w, ba.w);
	// -a.w - t * ba.w <= a.y + t * ba.y <= a.w + t * ba.w
	clip_range(-a.w,-ba.w, a.y, ba.y);
	clip_range( a.y, ba.y, a.w, ba.w);
	// -a.w - t * ba.w <= a.z + t * ba.z <= a.w + t * ba.w
	clip_range(-a.w,-ba.w, a.z, ba.z);
	clip_range( a.z, ba.z, a.w, ba.w);

	if (min_t < max_t) {
		if (min_t == 0.0f) {
			emit_vertex(va);
		} else {
			ShadedVertex out = lerp(va,vb,min_t);
			//don't interpolate attributes if in flat shading mode:
			if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Flat) out.attributes = va.attributes;
			emit_vertex(out);
		}
		if (max_t == 1.0f) {
			emit_vertex(vb);
		} else {
			ShadedVertex out = lerp(va,vb,max_t);
			//don't interpolate attributes if in flat shading mode:
			if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Flat) out.attributes = va.attributes;
			emit_vertex(out);
		}
	}
}


/*
 * clip_triangle - clip triangle to portion with -w <= x,y,z <= w, emit resulting shape as triangles (if non-empty)
 *  va, vb, vc: vertices of triangle
 *  emit_vertex: call to produce clipped triangles (three calls per triangle)
 *
 * If clipping truncates the triangle, attributes of the new vertices should respect the pipeline's interpolation mode.
 * 
 * If no portion of the triangle remains after clipping, emit_vertex will not be called.
 *
 * The clipped triangle(s) should have the same winding order as the full triangle.
 *
 */
template< PrimitiveType p, class P, uint32_t flags >
void Pipeline< p, P, flags >::clip_triangle(
		ShadedVertex const &va, ShadedVertex const &vb, ShadedVertex const &vc,
		std::function< void(ShadedVertex const &) > const &emit_vertex
	) {
	//A1EC: clip_triangle
	//TODO: correct code!
	emit_vertex(va);
	emit_vertex(vb);
	emit_vertex(vc);
}

template< PrimitiveType p, class P, uint32_t F >
bool Pipeline< p, P, F >::point_is_in_segment_range(
	float x, float y, float x1, float y1, float x2, float y2){
	if (x1 <= x && x <= x2){
		if (y1 <= y && y <= y2){
			return true;
		}else if (y2 <= y && y <= y1){
			return true;
		}
	}
	return false;
}

template< PrimitiveType p, class P, uint32_t F >
bool Pipeline< p, P, F >::segment_intersects_diamond(
	int px, int py, float x1, float y1, float x2, float y2){
	// special case: vertical segment
	if (x2 == x1 && y2 != y1){
		// first check x value
		if (static_cast<float>(px) <= x1 
			&& x1 < static_cast<float>(px) + 1.0f){
			return point_is_in_segment_range(x1, static_cast<float>(py) + 0.5f, 
			x1, y1, x2, y2);
		}
		return false;
	}

	// left to right direction
	if (x2 < x1){
		float x1t = x1;
		x1 = x2;
		x2 = x1t;
		float y1t = y1;
		y1 = y2;
		y2 = y1t;
	}
	// compute intercepts
	float m = (y2 - y1)/(x2 - x1);
	float b = y1 - x1 * m;
	float horizontal_intercept = (static_cast<float>(py) - b) / m;
	float vertical_intercept = m * static_cast<float>(px) + b;

	// if either intersects are within 0.5 from pixel center and
	// inside the segment, the segment intersects the pixel.
	if (static_cast<float>(px) <= horizontal_intercept
		&& horizontal_intercept < static_cast<float>(px) + 1.0f){
		// check if the intercept point is actually on the segment
		// since the intercept check only makes sure that the line of 
		// infinte length aligned to the segment intersects the diamond
		return point_is_in_segment_range(horizontal_intercept, static_cast<float>(py) + 0.5f, 
			x1, y1, x2, y2);
	}

	if (static_cast<float>(py) <= vertical_intercept
		&& vertical_intercept < static_cast<float>(py) + 1.0f){
		return point_is_in_segment_range(static_cast<float>(px) + 0.5f, vertical_intercept,
			x1, y1, x2, y2);
	}
	return false;
}

// helper function to implement diamond rule
// return values:
// 0: inside diamond
// 1: SW
// 2: SE
// 3: NW
// 4: NE
template< PrimitiveType p, class P, uint32_t F >
int Pipeline< p, P, F >::diamond_region(
	int px, int py, float x, float y){
	float x_center = static_cast<float>(px) + 0.5f;
	float y_center = static_cast<float>(py) + 0.5f;

	// specifically exclude top and right endpoint of diamond
	if (x == px + 1.0f && y == py + 0.5f){
		// right endpoint
		return E;
	}

	if (x == px + 0.5f && y == py + 1.0f){
		// top endpoint
		return N;
	}

	// check if inside diamond
	if (std::abs(x_center - x) + std::abs(y_center - y) <= 0.5f){
		return INSIDE_DIAMOND;
	}

	if (x <= x_center && y <= y_center){
		return SW;
	}

	if (x > x_center && y <= y_center){
		return SE;
	}
	
	if (x <= x_center && y > y_center){
		return NW;
	}
	return NE;
}


// draw backwards lines as forward by swapping begin and end pixel behavior
template< PrimitiveType p, class P, uint32_t flags >
void Pipeline< p, P, flags >::draw_line(int xa, int ya, int xb, int yb,
	bool emit_start_pixel,
	bool emit_end_pixel,
  	std::function< void(Fragment const &) > const &emit_fragment,
	ClippedVertex const &va, ClippedVertex const &vb){

	// swap endpoint behavior, as well the endpoints themselves
	// for a backwards line
	if (xb < xa){
		bool start_temp = emit_start_pixel;
		emit_start_pixel = emit_end_pixel;
		emit_end_pixel = start_temp;
		int x_temp = xa;
		int y_temp = ya;
		xa = xb;
		ya = yb;
		xb = x_temp;
		yb = y_temp;
	}

	float za = va.fb_position.z;
	float zb = vb.fb_position.z;
	float dist_ab = Vec2(xb - xa, yb - ya).norm();

	if (std::abs(yb - ya) <= xb - xa && ya <= yb){
		// positive slope x major
		int slope_err = 0 - (xb - xa);
		int y = ya;
		bool skip = false;
		for (int x = xa; x <= xb; x++) {
			skip = (x == xa && !emit_start_pixel) || (x == xb && !emit_end_pixel);
			Fragment f;
			f.fb_position.x = static_cast<float>(x) + 0.5f;
			f.fb_position.y = static_cast<float>(y) + 0.5f;
			f.fb_position.z = za + Vec2(f.fb_position.x - xa, f.fb_position.y - ya).norm() / dist_ab * (zb - za);
			f.attributes = va.attributes;
			// printf("xd%d, yd%d, xf%f, yf%f, skip%d\n", x, y, f.fb_position.x, f.fb_position.y, skip);
			if (!skip){
				f.derivatives.fill(Vec2(0.0f, 0.0f));
				emit_fragment(f);
			}

			slope_err += 2 * (yb - ya);
				
			if (slope_err > 0){
				y++;
				slope_err -= 2 * (xb - xa);
			}

		}
	} else if (std::abs(yb - ya) <= xb - xa && ya > yb){
		// negative slope x major
		int slope_err = 0 - (xb - xa);
		int y = ya;
		bool skip = false;
		for (int x = xa; x <= xb; x++) {
			skip = (x == xa && !emit_start_pixel) || (x == xb && !emit_end_pixel);
			Fragment f;
			f.fb_position.x = static_cast<float>(x) + 0.5f;
			f.fb_position.y = static_cast<float>(y) + 0.5f;
			f.fb_position.z = za + Vec2(f.fb_position.x - xa, f.fb_position.y - ya).norm() / dist_ab * (zb - za);
			f.attributes = va.attributes;
			// printf("xd%d, yd%d, xf%f, yf%f, skip%d\n", x, y, f.fb_position.x, f.fb_position.y, skip);
			if (!skip){
				f.derivatives.fill(Vec2(0.0f, 0.0f));
				emit_fragment(f);
			}

			slope_err += 2 * (ya - yb);
				
			if (slope_err > 0){
				y--;
				slope_err -= 2 * (xb - xa);
			}
		}
	} else if (ya <= yb) {
		// positive slope y major
		int slope_err = 0 - (yb - ya);
		int x = xa;
		bool skip = false;
		for (int y = ya; y <= yb; y++) {
			skip = (y == ya && !emit_start_pixel) || (y == yb && !emit_end_pixel);
			Fragment f;
			f.fb_position.x = static_cast<float>(x) + 0.5f;
			f.fb_position.y = static_cast<float>(y) + 0.5f;
			f.fb_position.z = za + Vec2(f.fb_position.x - xa, f.fb_position.y - ya).norm() / dist_ab * (zb - za);
			f.attributes = va.attributes;
			// printf("xd%d, yd%d, xf%f, yf%f, skip%d\n", x, y, f.fb_position.x, f.fb_position.y, skip);
			if (!skip){
				f.derivatives.fill(Vec2(0.0f, 0.0f));
				emit_fragment(f);
			}

			slope_err += 2 * (xb - xa);
				
			if (slope_err > 0){
				x++;
				slope_err -= 2 * (yb - ya);
			}

		}
	} else {
		// negative slope y major
		// printf("negative slope y major, xa%d, ya%d, xb%d, yb%d\n", xa, ya, xb, yb);
		int slope_err = 0 - (yb - ya);
		int x = xa;
		bool skip = false;
		for (int y = ya; y >= yb; y--) {
			skip = (y == ya && !emit_start_pixel) || (y == yb && !emit_end_pixel);
			Fragment f;
			f.fb_position.x = static_cast<float>(x) + 0.5f;
			f.fb_position.y = static_cast<float>(y) + 0.5f;
			f.fb_position.z = za + Vec2(f.fb_position.x - xa, f.fb_position.y - ya).norm() / dist_ab * (zb - za);
			f.attributes = va.attributes;
			// printf("xd%d, yd%d, xf%f, yf%f, skip%d\n", x, y, f.fb_position.x, f.fb_position.y, skip);
			if (!skip){
				f.derivatives.fill(Vec2(0.0f, 0.0f));
				emit_fragment(f);
			}
			slope_err += 2 * (xb - xa);
				
			if (slope_err > 0){
				x++;
				slope_err += 2 * (yb - ya);
			}
		}
	}


}




//-------------------------------------------------------------------------
//rasterization functions

/*
 * rasterize_line:
 * calls emit_fragment( frag ) for every pixel "covered" by the line (va.fb_position.xy, vb.fb_position.xy).
 *
 *    a pixel (x,y) is "covered" by the line if it exits the inscribed diamond:
 * 
 *        (x+0.5,y+1)
 *        /        \
 *    (x,y+0.5)  (x+1,y+0.5)
 *        \        /
 *         (x+0.5,y)
 *
 *    to avoid ambiguity, we consider diamonds to contain their left and bottom points
 *    but not their top and right points. 
 * 
 * 	  since 45 degree lines breaks this rule, our rule in general is to rasterize the line as if its
 *    endpoints va and vb were at va + (e, e^2) and vb + (e, e^2) where no smaller nonzero e produces 
 *    a different rasterization result.
 *
 * for each such diamond, pass Fragment frag to emit_fragment, with:
 *  - frag.fb_position.xy set to the center (x+0.5,y+0.5)
 *  - frag.fb_position.z interpolated linearly between va.fb_position.z and vb.fb_position.z
 *  - frag.attributes set to va.attributes (line will only be used in Interp_Flat mode)
 *  - frag.derivatives set to all (0,0)
 *
 * when interpolating the depth (z) for the fragments, you may use any depth the line takes within the pixel
 * (i.e., you don't need to interpolate to, say, the closest point to the pixel center)
 *
 * If you wish to work in fixed point, check framebuffer.h for useful information about the framebuffer's dimensions.
 *
 */

template< PrimitiveType p, class P, uint32_t flags >
void Pipeline< p, P, flags >::rasterize_line(
		ClippedVertex const &va, ClippedVertex const &vb,
		std::function< void(Fragment const &) > const &emit_fragment
	) {
	if constexpr ((flags & PipelineMask_Interp) != Pipeline_Interp_Flat) {
		assert(0 && "rasterize_line should only be invoked in flat interpolation mode.");
	}
	//A1T2: rasterize_line
	
	//TODO: Check out the block comment above this function for more information on how to fill in this function!
	// 		The OpenGL specification section 3.5 may also come in handy.

	// { //As a placeholder, draw a point in the middle of the line:
	// 	//(remove this code once you have a real implementation)
	// 	Fragment mid;
	// 	mid.fb_position = (va.fb_position + vb.fb_position) / 2.0f;
	// 	mid.attributes = va.attributes;
	// 	mid.derivatives.fill(Vec2(0.0f, 0.0f));
	// 	emit_fragment(mid); 
	// }

	float xa_f = va.fb_position.x;
	float xb_f = vb.fb_position.x;
	float ya_f = va.fb_position.y;
	float yb_f = vb.fb_position.y;
	int xa = static_cast<int>(xa_f);
	int ya = static_cast<int>(ya_f);
	int xb = static_cast<int>(xb_f);
	int yb = static_cast<int>(yb_f);

	// printf("xa%d, ya%d, xb%d, yb%d\n", xa, ya, xb, yb);

	bool emit_start_pixel = false;
	bool emit_end_pixel = true;

	// apply diamond exit rule for first pixel
	if (diamond_region(xa, ya, xa_f, ya_f) == INSIDE_DIAMOND
		&& diamond_region(xa, ya, xb_f, yb_f) != INSIDE_DIAMOND){
		// if the first pixel is inside the diamond 
		// and the last pixel is not inside the same diamond
		// we can emit the 1st pixel.
		emit_start_pixel = true;
		// printf("condition 1\n");
	}else if (segment_intersects_diamond(xa, ya, xa_f, ya_f, xb_f, yb_f)){
		// if the first pixel is not in the diamond,
		// emit if the line segment intersects the 
		// diamond
		// printf("condition 2\n");
		emit_start_pixel = true;
	}

	// apply diamond exit rule for last pixel
	if (diamond_region(xb, yb, xb_f, yb_f) == INSIDE_DIAMOND){
		// if the last pixel is in its diamond, dont emit
		// since the segment doesnt exit that diamond
		emit_end_pixel = false;
		// printf("condition 3\n");
	}else if (!segment_intersects_diamond(xb, yb, xa_f, ya_f, xb_f, yb_f)){
		// if the last pixel is not in its diamond, dont emit
		// if the segment doesnt intersect that diamond
		emit_end_pixel = false;
		// printf("condition 4\n");
	}

	// printf("s%d, e%d\n", emit_start_pixel, emit_end_pixel);

	draw_line(xa, ya, xb, yb, emit_start_pixel, emit_end_pixel, emit_fragment, va, vb);
}

// use Heron's formula to calculate the area of a triangle
// perform calculations in long double to mitigate loss
// of precision
template< PrimitiveType p, class P, uint32_t flags >
float Pipeline< p, P, flags >::area_triangle(Vec2 va, Vec2 vb, Vec2 vc){
	long double a = static_cast<long double>((va - vb).norm());
	long double b = static_cast<long double>((va - vc).norm());
	long double c = static_cast<long double>((vb - vc).norm());
	long double s = ((a + b + c) / 2.0L);
	return static_cast<float>(std::sqrt(s * (s - a) * (s - b) * (s - c)));
}

// Compute Barycentric coordinates
template< PrimitiveType p, class P, uint32_t flags >
Vec3 Pipeline< p, P, flags >::barycentric(Vec2 vp, Vec2 va, Vec2 vb, Vec2 vc){
	float denom = (det(va, vb) + det(vb, vc) + det(vc, va));
	float wa = (det(vb, vc) + det(vp, (vb - vc)))/ denom;
	float wb = (det(vc, va) + det(vp, (vc - va)))/ denom;
	float wc = (det(va, vb) + det(vp, (va - vb)))/ denom;
	return Vec3(wa, wb, wc);
}

// interpolate
template< PrimitiveType p, class P, uint32_t flags >
float Pipeline< p, P, flags >::interp_triangle(Vec3 traits, Vec2 vp, Vec2 va, Vec2 vb, Vec2 vc){
	float denom = (det(va, vb) + det(vb, vc) + det(vc, va));
	float wa = (det(vb, vc) + det(vp, (vb - vc)))/ denom;
	float wb = (det(vc, va) + det(vp, (vc - va)))/ denom;
	float wc = (det(va, vb) + det(vp, (va - vb)))/ denom;
	return dot(Vec3(wa, wb, wc), traits);
}

template< PrimitiveType p, class P, uint32_t flags >
bool Pipeline< p, P, flags >::in_triangle(int x, int y,
	ClippedVertex const &va, ClippedVertex const &vb, ClippedVertex const &vc){
	// conduct a halfplane check on the pixel center at (x+0.5, y+0.5)
	// start at the leftmost vertex. The two sides exiting this vertex
	// are guaranteed to be top or left. Therefore, conduct inclusive
	// halfplane check on these two sides. Then conduct exclusive halfplane
	// check on the third side.
	float x_center = x + 0.5f;
	float y_center = y + 0.5f;

	float x_min = std::min({va.fb_position.x, vb.fb_position.x, vc.fb_position.x});
	// float y_min = std::min({va.fb_position.y, vb.fb_position.y, vc.fb_position.y});
	ClippedVertex vleft, v1, v2;
	if (va.fb_position.x == x_min){
		vleft = va;
		v1 = vb;
		v2 = vc;
	} else if (vb.fb_position.x == x_min){
		vleft = vb;
		v1 = va;
		v2 = vc;
	} else {
		vleft = vc;
		v1 = va;
		v2 = vb;
	}

	float m1 = (v1.fb_position.y - vleft.fb_position.y)/(v1.fb_position.x - vleft.fb_position.x);
	float b1 = vleft.fb_position.y - vleft.fb_position.x * m1;

	float m2 = (v2.fb_position.y - vleft.fb_position.y)/(v2.fb_position.x - vleft.fb_position.x);
	float b2 = vleft.fb_position.y - vleft.fb_position.x * m2;

	float m3 = (v2.fb_position.y - v1.fb_position.y)/(v2.fb_position.x - v1.fb_position.x);
	float b3 = v1.fb_position.y - v1.fb_position.x * m3;

	// sides 1 and 2 are connected to the leftmost edge.
	// both side1 and side2 are inclusively checked,
	// since they must either be left edge or top edge
	// check side1 = vleft -> v1
	// check vertical special case
	if (v1.fb_position.x == vleft.fb_position.x){
		if (x_center < v1.fb_position.x)
			return false;
	} else {
		if (v1.fb_position.y >= v2.fb_position.y && y_center > m1 * x_center + b1)
			return false;
		if (v1.fb_position.y < v2.fb_position.y && y_center < m1 * x_center + b1)
			return false;
	}

	// check side vleft -> v2
	// check vertical special case
	if (v2.fb_position.x == vleft.fb_position.x){
		if (x_center < v2.fb_position.x)
			return false;
	} else {
		if (v2.fb_position.y >= v1.fb_position.y && y_center > m2 * x_center + b2)
			return false;
		if (v2.fb_position.y < v1.fb_position.y && y_center < m2 * x_center + b2)
			return false;
	}

	// check side v1 -> v2
	// check vertical special case
	if (v2.fb_position.x == v1.fb_position.x){
		if (x_center >= v2.fb_position.x)
			return false;
	} else {
		if (v2.fb_position.y >= vleft.fb_position.y && v1.fb_position.y >= vleft.fb_position.y) {
			if (y_center >= m3 * x_center + b3)
				return false;
		} else if (v2.fb_position.y <= vleft.fb_position.y && v1.fb_position.y <= vleft.fb_position.y) {
			if (y_center <= m3 * x_center + b3)
				return false;
		}

		if (m3 >= 0 && y_center <= m3 * x_center + b3)
			return false;
		if (m3 < 0 && y_center >= m3 * x_center + b3)
			return false;
		
	}

	return true;
}


/*
 *
 * rasterize_triangle(a,b,c,emit) calls 'emit(frag)' at every location
 *  (x+0.5,y+0.5) (where x,y are integers) covered by triangle (a,b,c).
 *
 * The emitted fragment should have:
 * - frag.fb_position.xy = (x+0.5, y+0.5)
 * - frag.fb_position.z = linearly interpolated fb_position.z from a,b,c (NOTE: does not depend on Interp mode!)
 * - frag.attributes = depends on Interp_* flag in flags:
 *   - if Interp_Flat: copy from va.attributes
 *   - if Interp_Screen: interpolate as if (a,b,c) is a 2D triangle flat on the screen
 *   - if Interp_Correct: use perspective-correct interpolation
 * - frag.derivatives = derivatives w.r.t. fb_position.x and fb_position.y of the first frag.derivatives.size() attributes.
 *
 * Notes on derivatives:
 *  The derivatives are partial derivatives w.r.t. screen locations. That is:
 *    derivatives[i].x = d/d(fb_position.x) attributes[i]
 *    derivatives[i].y = d/d(fb_position.y) attributes[i]
 *  You may compute these derivatives analytically or numerically.
 *
 *  See section 8.12.1 "Derivative Functions" of the GLSL 4.20 specification for some inspiration. (*HOWEVER*, the spec is solving a harder problem, and also nothing in the spec is binding on your implementation)
 *
 *  One approach is to rasterize blocks of four fragments and use forward and backward differences to compute derivatives.
 *  To assist you in this approach, keep in mind that the framebuffer size is *guaranteed* to be even. (see framebuffer.h)
 *
 * Notes on coverage:
 *  If two triangles are on opposite sides of the same edge, and a
 *  fragment center lies on that edge, rasterize_triangle should
 *  make sure that exactly one of the triangles emits that fragment.
 *  (Otherwise, speckles or cracks can appear in the final render.)
 * 
 *  For degenerate (co-linear) triangles, you may consider them to not be on any side of an edge.
 * 	Thus, even if two degnerate triangles share an edge that contains a fragment center, you don't need to emit it.
 *  You will not lose points for doing something reasonable when handling this case
 *
 *  This is pretty tricky to get exactly right!
 *
 */
template< PrimitiveType p, class P, uint32_t flags >
void Pipeline< p, P, flags >::rasterize_triangle(
		ClippedVertex const &va, ClippedVertex const &vb, ClippedVertex const &vc,
		std::function< void(Fragment const &) > const &emit_fragment
	) {
	//NOTE: it is okay to restructure this function to allow these tasks to use the
	// same code paths. Be aware, however, that all of them need to remain working!
	// (e.g., if you break Flat while implementing Correct, you won't get points
	//  for Flat.)
	if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Flat) {
		// step 0: check if the triangle is degenerate. 
		if (area_triangle(va.fb_position.xy(), vb.fb_position.xy(), vc.fb_position.xy()) == 0)
			return;

		// step 1: generate screeen space bounding box from the 3 vertices
		int xmin = static_cast<int>(std::floor(std::min(va.fb_position.x, std::min(vb.fb_position.x, vc.fb_position.x))));
		int xmax = static_cast<int>(std::ceil(std::max(va.fb_position.x, std::max(vb.fb_position.x, vc.fb_position.x))));
		int ymin = static_cast<int>(std::floor(std::min(va.fb_position.y, std::min(vb.fb_position.y, vc.fb_position.y))));
		int ymax = static_cast<int>(std::ceil(std::max(va.fb_position.y, std::max(vb.fb_position.y, vc.fb_position.y))));

		float za = va.fb_position.z;
		float zb = vb.fb_position.z;
		float zc = vc.fb_position.z;
		bool same_z = false;
		ClippedVertex vj, vk;
		if (za == zb && zb == zc){
			same_z = true;
		}

		// step 2: iterate over all pixels inside bounding box, conducting a coverage
		// test for each pixel. 

		// step3: for each pixel passing coverage test, z value is interpolated using barycentric coorhinates

		for (int x = xmin; x <= xmax; x++){
			for (int y = ymin; y <= ymax; y++){
				if (in_triangle(x, y, va, vb, vc)){
					float xf = static_cast<float>(x) + 0.5f;
					float yf = static_cast<float>(y) + 0.5f;
					float zf;
					if (same_z){
						zf = za;
					} else {
						zf = interp_triangle(
							Vec3(va.fb_position.z, vb.fb_position.z, vc.fb_position.z),
							Vec2(xf, yf),
							va.fb_position.xy(), 
							vb.fb_position.xy(),
							vc.fb_position.xy());
					}
					Fragment f;
					f.fb_position.x = xf;
					f.fb_position.y = yf;
					f.fb_position.z = zf;
					f.attributes = va.attributes;
					f.derivatives.fill(Vec2(0.0f, 0.0f));
					emit_fragment(f);
				}
			}
		}
	} else if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Screen) {
		// step 0: check if the triangle is degenerate. 
		if (area_triangle(va.fb_position.xy(), vb.fb_position.xy(), vc.fb_position.xy()) == 0)
			return;

		// step 1: generate screeen space bounding box from the 3 vertices
		int xmin = static_cast<int>(std::floor(std::min(va.fb_position.x, std::min(vb.fb_position.x, vc.fb_position.x))));
		int xmax = static_cast<int>(std::ceil(std::max(va.fb_position.x, std::max(vb.fb_position.x, vc.fb_position.x))));
		int ymin = static_cast<int>(std::floor(std::min(va.fb_position.y, std::min(vb.fb_position.y, vc.fb_position.y))));
		int ymax = static_cast<int>(std::ceil(std::max(va.fb_position.y, std::max(vb.fb_position.y, vc.fb_position.y))));

		float za = va.fb_position.z;
		float zb = vb.fb_position.z;
		float zc = vc.fb_position.z;
		bool same_z = false;
		ClippedVertex vj, vk;
		if (za == zb && zb == zc){
			same_z = true;
		}
		const int VA_TexCoordU = 0;
		const int VA_TexCoordV = 1;

		// step 2: iterate over all pixels inside bounding box, conducting a coverage
		// test for each pixel. 

		// step3: for each pixel passing coverage test, z value is interpolated using barycentric coordinates, and so are
		// attributes.

		for (int x = xmin; x <= xmax; x++){
			for (int y = ymin; y <= ymax; y++){
				if (in_triangle(x, y, va, vb, vc)){
					float xf = static_cast<float>(x) + 0.5f;
					float yf = static_cast<float>(y) + 0.5f;
					float zf;
					
					if (same_z){
						zf = za;
					} else {
						zf = interp_triangle(
							Vec3(va.fb_position.z, vb.fb_position.z, vc.fb_position.z),
							Vec2(xf, yf),
							va.fb_position.xy(), 
							vb.fb_position.xy(),
							vc.fb_position.xy());
					}
					Fragment f;
					f.fb_position.x = xf;
					f.fb_position.y = yf;
					f.fb_position.z = zf;

					for (int i = 0; i < 5; i++){
						f.attributes[i] =
							interp_triangle(
							Vec3(va.attributes[i], vb.attributes[i], vc.attributes[i]),
							Vec2(xf, yf),
							va.fb_position.xy(), 
							vb.fb_position.xy(),
							vc.fb_position.xy());
					}

					float x_plus1_u = interp_triangle(
						Vec3(va.attributes[VA_TexCoordU], vb.attributes[VA_TexCoordU], vc.attributes[VA_TexCoordU]),
						Vec2(xf + 1.0f, yf),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy());
					float x_plus1_v = interp_triangle(
						Vec3(va.attributes[VA_TexCoordV], vb.attributes[VA_TexCoordV], vc.attributes[VA_TexCoordV]),
						Vec2(xf + 1.0f, yf),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy());
					float y_plus1_u = interp_triangle(
						Vec3(va.attributes[VA_TexCoordU], vb.attributes[VA_TexCoordU], vc.attributes[VA_TexCoordU]),
						Vec2(xf, yf + 1.0f),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy());
					float y_plus1_v = interp_triangle(
						Vec3(va.attributes[VA_TexCoordV], vb.attributes[VA_TexCoordV], vc.attributes[VA_TexCoordV]),
						Vec2(xf, yf + 1.0f),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy());

					float dUdx = x_plus1_u - f.attributes[0];
					float dVdx = x_plus1_v - f.attributes[1];
					
					float dUdy = y_plus1_u - f.attributes[0];
					float dVdy = y_plus1_v - f.attributes[1];

					f.derivatives[0] = Vec2(dUdx, dUdy);
					f.derivatives[1] = Vec2(dVdx, dVdy);

					emit_fragment(f);
				}
			}
		}


	} else if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Correct) {
		// step 0: check if the triangle is degenerate. 
		if (area_triangle(va.fb_position.xy(), vb.fb_position.xy(), vc.fb_position.xy()) == 0)
			return;

		// step 1: generate screeen space bounding box from the 3 vertices
		int xmin = static_cast<int>(std::floor(std::min(va.fb_position.x, std::min(vb.fb_position.x, vc.fb_position.x))));
		int xmax = static_cast<int>(std::ceil(std::max(va.fb_position.x, std::max(vb.fb_position.x, vc.fb_position.x))));
		int ymin = static_cast<int>(std::floor(std::min(va.fb_position.y, std::min(vb.fb_position.y, vc.fb_position.y))));
		int ymax = static_cast<int>(std::ceil(std::max(va.fb_position.y, std::max(vb.fb_position.y, vc.fb_position.y))));

		float za = va.fb_position.z;
		float zb = vb.fb_position.z;
		float zc = vc.fb_position.z;
		bool same_z = false;
		ClippedVertex vj, vk;
		if (za == zb && zb == zc){
			same_z = true;
		}
		const int VA_TexCoordU = 0;
		const int VA_TexCoordV = 1;

		// step 2: iterate over all pixels inside bounding box, conducting a coverage
		// test for each pixel. 

		// step3: for each pixel passing coverage test, z value is interpolated using barycentric coordinates, and so are
		// attributes.

		for (int x = xmin; x <= xmax; x++){
			for (int y = ymin; y <= ymax; y++){
				if (in_triangle(x, y, va, vb, vc)){
					float xf = static_cast<float>(x) + 0.5f;
					float yf = static_cast<float>(y) + 0.5f;
					float zf;
					
					if (same_z){
						zf = za;
					} else {
						zf = interp_triangle(
							Vec3(va.fb_position.z, vb.fb_position.z, vc.fb_position.z),
							Vec2(xf, yf),
							va.fb_position.xy(), 
							vb.fb_position.xy(),
							vc.fb_position.xy());
					}
					Fragment f;
					f.fb_position.x = xf;
					f.fb_position.y = yf;
					f.fb_position.z = zf;

					// interp phi over w
					float inv_w = interp_triangle(
							Vec3(va.inv_w, vb.inv_w, vc.inv_w),
							Vec2(xf, yf),
							va.fb_position.xy(), 
							vb.fb_position.xy(),
							vc.fb_position.xy());

					for (int i = 0; i < 5; i++){
						f.attributes[i] =
							interp_triangle(
							Vec3(va.attributes[i]*va.inv_w, vb.attributes[i]*vb.inv_w, vc.attributes[i]*vc.inv_w),
							Vec2(xf, yf),
							va.fb_position.xy(), 
							vb.fb_position.xy(),
							vc.fb_position.xy()) / inv_w;
					}

					// derivatives
					float inv_w_xplus1 = interp_triangle(
						Vec3(va.inv_w, vb.inv_w, vc.inv_w),
						Vec2(xf + 1.0f, yf),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy());

					float inv_w_yplus1 = interp_triangle(
						Vec3(va.inv_w, vb.inv_w, vc.inv_w),
						Vec2(xf, yf + 1.0f),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy());

					float x_plus1_u = interp_triangle(
						Vec3(va.attributes[VA_TexCoordU]*va.inv_w,
						 	vb.attributes[VA_TexCoordU]*vb.inv_w,
						  	vc.attributes[VA_TexCoordU]*vb.inv_w),
						Vec2(xf + 1.0f, yf),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy())/inv_w_xplus1;
					float x_plus1_v = interp_triangle(
						Vec3(va.attributes[VA_TexCoordV]*va.inv_w, 
							vb.attributes[VA_TexCoordV]*vb.inv_w,
							vc.attributes[VA_TexCoordV]*vb.inv_w),
						Vec2(xf + 1.0f, yf),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy())/inv_w_xplus1;
					float y_plus1_u = interp_triangle(
						Vec3(va.attributes[VA_TexCoordU]*va.inv_w,
							vb.attributes[VA_TexCoordU]*vb.inv_w,
							vc.attributes[VA_TexCoordU]*vb.inv_w),
						Vec2(xf, yf + 1.0f),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy())/inv_w_yplus1;
					float y_plus1_v = interp_triangle(
						Vec3(va.attributes[VA_TexCoordV]*va.inv_w, 
							vb.attributes[VA_TexCoordV]*vb.inv_w, 
							vc.attributes[VA_TexCoordV]*vb.inv_w),
						Vec2(xf, yf + 1.0f),
						va.fb_position.xy(), 
						vb.fb_position.xy(),
						vc.fb_position.xy())/inv_w_yplus1;

					float dUdx = x_plus1_u - f.attributes[0];
					float dVdx = x_plus1_v - f.attributes[1];
					
					float dUdy = y_plus1_u - f.attributes[0];
					float dVdy = y_plus1_v - f.attributes[1];

					f.derivatives[0] = Vec2(dUdx, dUdy);
					f.derivatives[1] = Vec2(dVdx, dVdy);

					emit_fragment(f);
				}
			}
		}
	}
}


//-------------------------------------------------------------------------
//compile instantiations for all programs and blending and testing types:

#include "programs.h"

template struct Pipeline< PrimitiveType::Lines, Programs::Lambertian, Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Flat >;
template struct Pipeline< PrimitiveType::Lines, Programs::Lambertian, Pipeline_Blend_Replace | Pipeline_Depth_Always | Pipeline_Interp_Flat >;
template struct Pipeline< PrimitiveType::Triangles, Programs::Lambertian, Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Flat >;
template struct Pipeline< PrimitiveType::Triangles, Programs::Lambertian, Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Screen >;
template struct Pipeline< PrimitiveType::Triangles, Programs::Lambertian, Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Correct >;
