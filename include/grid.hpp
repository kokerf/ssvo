#ifndef _SSVO_GRID_HPP_
#define _SSVO_GRID_HPP_

#include "global.hpp"

namespace ssvo
{

template<typename T>
class Grid
{
public:
    typedef std::list<T> Cell;
    typedef std::vector<std::shared_ptr<Cell> > Cells;

    Grid(size_t cols, size_t rows, size_t size) :
        cols_(cols), rows_(rows), area_(cols * rows), grid_size_(size),
        grid_n_cols_(0), grid_n_rows_(0), grid_n_cells_(0),
        cells_(std::make_shared<Cells>(Cells()))
    {
        reset(grid_size_);
    }

    void reset(size_t grid_size)
    {
        grid_size_ = grid_size;
        grid_n_cols_ = ceil(static_cast<double>(cols_) / grid_size_);
        grid_n_rows_ = ceil(static_cast<double>(rows_) / grid_size_);
        grid_n_cells_ = grid_n_cols_ * grid_n_rows_;

        cells_.reset(new Cells(grid_n_cells_));
        for(std::shared_ptr<Cell> &cell : *cells_)
        { cell = std::make_shared<Cell>(Cell()); }

        mask_.clear();
        mask_.resize(grid_n_cells_, false);
    }

    void clear()
    {
        for(std::shared_ptr<Cell> &cell : *cells_)
        { cell->clear(); }
        std::fill(mask_.begin(), mask_.end(), false);
    }

    void sort()
    {
        for(std::shared_ptr<Cell> &cell : *cells_)
        { cell->sort(); }
    }

    size_t size()
    {
        return (size_t) std::count_if(cells_->begin(), cells_->end(), [](const std::shared_ptr<Cell> &cell) { return !cell->empty(); });
    }

    void resize(size_t grid_size)
    {
        if(grid_size == grid_size_)
            return;
        std::shared_ptr<Cells> old_cells = cells_;
        reset(grid_size);
        for(std::shared_ptr<Cell> &cell : *old_cells)
            for(const T &element : *cell)
                insert(element);
    }

    inline size_t insert(const T &element)
    {
        const size_t id = getIndex(element);
        if(mask_.at(id))
            return 0;
        const std::shared_ptr<Cell> &cell = cells_->at(id);
        cell->push_back(element);
        return (size_t) cell->size();
    }

    inline size_t remove(const T &element)
    {
        const size_t id = getIndex(element);
        if(mask_.at(id))
            return 0;
        const std::shared_ptr<Cell> &cell = cells_->at(id);
        cell->remove(element);
        return (size_t) cell->size();
    }

    size_t getIndex(const T &element)
    {
        std::cerr << "Do not use the function[ size_t getCellID(T &element) ]! Please Specialized!" << std::endl;
        std::abort();
    }

    void getBestElement(Cell &out)
    {
        out.clear();
        for(size_t idx = 0; idx < grid_n_cells_; idx++)
        {
            if(mask_.at(idx))
                continue;
            std::shared_ptr<Cell> &cell = cells_->at(idx);
            out.push_back(*cell->rbegin());
            if(!cell->empty())
                out.push_back(*std::max_element(cell->begin(), cell->end()));
        }
    }

    void getBestElement(std::vector<T> &out)
    {
        out.clear();
        out.reserve(size());
        for(size_t idx = 0; idx < grid_n_cells_; idx++)
        {
            if(mask_.at(idx))
                continue;
            std::shared_ptr<Cell> &cell = cells_->at(idx);
            if(!cell->empty())
                out.push_back(*std::max_element(cell->begin(), cell->end()));
        }
    }

    void setMask(const std::vector<bool> &mask)
    {
        assert(mask.size() == grid_n_cells_);
        mask_ = mask;
    }

    void setMask(size_t index, bool value = true)
    {
        assert(index >= 0 && index < grid_n_cells_);
        mask_.at(index) = value;
    }

    inline const size_t cols()
    {
        return cols_;
    }

    inline const size_t rows()
    {
        return rows_;
    }

    inline const size_t area()
    {
        return area_;
    }

    inline const size_t nCells()
    {
        return grid_n_cells_;
    }

    inline const size_t gridSize()
    {
        return grid_size_;
    }

    inline const Cell &getCell(size_t id)
    {
        return *cells_->at(id);
    }

    inline const Cells &getCells()
    {
        return *cells_;
    }

private:

    const size_t cols_;
    const size_t rows_;
    const size_t area_;
    size_t grid_size_;
    size_t grid_n_cols_;
    size_t grid_n_rows_;
    size_t grid_n_cells_;

    std::shared_ptr<Cells> cells_;
    std::vector<bool> mask_;
};

}

#endif //_SSVO_GRID_HPP_
