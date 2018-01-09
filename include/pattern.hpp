#ifndef _SSVO_PATTERN_HPP_
#define _SSVO_PATTERN_HPP_

namespace ssvo{

template <typename T, int N, int S>
struct Pattern
{
    enum{Num = N, Size = S, Size1 = S+2};
    const std::array<std::array<int, 2>, N> data;
    std::array<std::array<int, 5>, N> offset;
//    std::array<std::array<bool, Size1>, Size1> mask;

    inline void getPattern(Matrix<T, Size1, Size1, RowMajor> &mat, Matrix<T, N, 3, RowMajor> &pattern) const
    {
        Matrix<T, 1, 3, RowMajor> *idxy = (Matrix<T, 1, 3, RowMajor>*) pattern.data();
        const T* mat_ptr = mat.data();
        for(int i = 0; i < N; i++, idxy++)
        {
            const std::array<int, 5> &ofs = offset[i];
            (*idxy)[0] = mat_ptr[ofs[0]];
            (*idxy)[1] = (mat_ptr[ofs[1]]-mat_ptr[ofs[2]])*0.5;
            (*idxy)[2] = (mat_ptr[ofs[3]]-mat_ptr[ofs[4]])*0.5;
        }
    }

    inline void getPattern(Matrix<T, Size1, Size1, RowMajor> &mat, Matrix<T, N, 1> &pattern) const
    {
        T *p = pattern.data();
        const T* mat_ptr = mat.data();
        for(int i = 0; i < N; i++, p++)
        {
            *p = mat_ptr[offset[i][0]];
        }
    }

    Pattern(std::array<std::array<int, 2>, N> array):
        data(array)
    {
        for(int i = 0; i < N; ++i)
        {
            assert(abs(data[i][0]) <= S && abs(data[i][1]) <= S);
            int index = (data[i][0]+Size1/2) + (data[i][1]+Size1/2) * Size1;
            offset[i] = {index, index+1, index-1, index+Size1, index-Size1};
//            bool* mask_ptr = mask.data()->data();
//            for(int p : offset[i])
//                mask_ptr[p] = true;
        }
    }
};

const Pattern<float, 64, 8> pattern0(
    {{
         {-4, -4}, {-3, -4}, {-2, -4}, {-1, -4}, { 0, -4}, { 1, -4}, { 2, -4}, { 3, -4},
         {-4, -3}, {-3, -3}, {-2, -3}, {-1, -3}, { 0, -3}, { 1, -3}, { 2, -3}, { 3, -3},
         {-4, -2}, {-3, -2}, {-2, -2}, {-1, -2}, { 0, -2}, { 1, -2}, { 2, -2}, { 3, -2},
         {-4, -1}, {-3, -1}, {-2, -1}, {-1, -1}, { 0, -1}, { 1, -1}, { 2, -1}, { 3, -1},
         {-4,  0}, {-3,  0}, {-2,  0}, {-1,  0}, { 0,  0}, { 1,  0}, { 2,  0}, { 3,  0},
         {-4,  1}, {-3,  1}, {-2,  1}, {-1,  1}, { 0,  1}, { 1,  1}, { 2,  1}, { 3,  1},
         {-4,  2}, {-3,  2}, {-2,  2}, {-1,  2}, { 0,  2}, { 1,  2}, { 2,  2}, { 3,  2},
         {-4,  3}, {-3,  3}, {-2,  3}, {-1,  3}, { 0,  3}, { 1,  3}, { 2,  3}, { 3,  3},
     }}
);

const Pattern<float, 16, 7> pattern1(
    {{
         {-1, -3}, { 0, -3}, { 1, -3},
         {-2, -2}, { 2, -2},
         {-3, -1}, { 3, -1},
         {-3,  0}, { 3,  0},
         {-3,  1}, { 3,  1},
         {-2,  2}, { 2,  2},
         {-1,  3}, { 0,  3}, { 1,  3},
     }}
);

const Pattern<float, 25, 7> pattern2(
    {{
         {-3, -3}, {-1, -3}, { 0, -3}, { 1, -3}, { 3, -3},
         {-2, -2}, { 2, -2},
         {-3, -1}, {-1, -1}, { 1, -1}, { 3, -1},
         {-3,  0}, { 0,  0}, { 3,  0},
         {-3,  1}, {-1,  1}, { 1,  1}, { 3,  1},
         {-2,  2}, { 2,  2},
         {-3,  3}, {-1,  3}, { 0,  3}, { 1,  3}, { 3,  3},
     }}
);

const Pattern<float, 31, 7> pattern3(
    {{
         {-1, -3}, { 0, -3}, { 1, -3},
         {-2, -2}, { 0, -2}, { 2, -2},
         {-3, -1}, { 0, -1}, { 3, -1},
         {-3,  0}, {-2,  0}, {-1,  0}, { 0,  0}, { 1,  0}, { 2,  0}, { 3,  0},
         {-3,  1}, { 0,  1}, { 3,  1},
         {-2,  2}, { 0,  2}, { 2,  2},
         {-1,  3}, { 0,  3}, { 1,  3},
     }}
);

const Pattern<float, 32, 8> pattern4(
    {{
         {-4, -4}, {-2, -4}, { 0, -4}, { 2, -4},
         {-3, -3}, {-1, -3}, { 1, -3}, { 3, -3},
         {-4, -2}, {-2, -2}, { 0, -2}, { 2, -2},
         {-3, -1}, {-1, -1}, { 1, -1}, { 3, -1},
         {-4,  0}, {-2,  0}, { 0,  0}, { 2,  0},
         {-3,  1}, {-1,  1}, { 1,  1}, { 3,  1},
         {-4,  2}, {-2,  2}, { 0,  2}, { 2,  2},
         {-3,  3}, {-1,  3}, { 1,  3}, { 3,  3},
     }}
);

const Pattern<float, 49, 13> pattern5(
    {{
         {-6, -6}, {-4, -6}, {-2, -6}, { 0, -6}, { 2, -6}, { 4, -6}, { 6, -6},
         {-6, -4}, {-4, -4}, {-2, -4}, { 0, -4}, { 2, -4}, { 4, -4}, { 6, -4},
         {-6, -2}, {-4, -2}, {-2, -2}, { 0, -2}, { 2, -2}, { 4, -2}, { 6, -2},
         {-6,  0}, {-4,  0}, {-2,  0}, { 0,  0}, { 2,  0}, { 4,  0}, { 6,  0},
         {-6,  2}, {-4,  2}, {-2,  2}, { 0,  2}, { 2,  2}, { 4,  2}, { 6,  2},
         {-6,  4}, {-4,  4}, {-2,  4}, { 0,  4}, { 2,  4}, { 4,  4}, { 6,  4},
         {-6,  6}, {-4,  6}, {-2,  6}, { 0,  6}, { 2,  6}, { 4,  6}, { 6,  6},
     }}
);

}

#endif //_SSVO_PATTERN_HPP_
